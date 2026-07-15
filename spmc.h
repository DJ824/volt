#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SPMC_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SPMC_ALWAYS_INLINE __forceinline
#else
#define SPMC_ALWAYS_INLINE inline
#endif

template<typename T, std::size_t SIZE, bool PAD_SLOTS = false, std::size_t RECLAIM_SCAN_LIMIT = 32>
class LockFreeQueueSpmcSeq
{
   static_assert(SIZE > 0, "SIZE must be non-zero");
   static_assert((SIZE & (SIZE - 1)) == 0, "SIZE must be a power of two");
   static_assert((CACHE_LINE_SIZE & (CACHE_LINE_SIZE - 1)) == 0, "CACHE_LINE_SIZE must be a power of two");
   static_assert(RECLAIM_SCAN_LIMIT > 0, "RECLAIM_SCAN_LIMIT must be non-zero");
   static_assert(std::atomic<std::size_t>::is_always_lock_free, "size_t atomics must be lock-free");
   static_assert(std::is_nothrow_move_assignable_v<T>, "T must be nothrow move assignable for pop(T&)");
   static_assert(std::is_nothrow_destructible_v<T>, "T must be nothrow destructible");

   static constexpr std::size_t CAPACITY = SIZE;
   static constexpr std::size_t MASK     = CAPACITY - 1;

   using AtomicSequence = std::atomic<std::size_t>;

   static constexpr std::size_t NATURAL_SLOT_ALIGNMENT = alignof(T) > alignof(AtomicSequence) ? alignof(T) :
                                                                                                alignof(AtomicSequence);

   static constexpr std::size_t SLOT_ALIGNMENT =
      PAD_SLOTS && CACHE_LINE_SIZE > NATURAL_SLOT_ALIGNMENT ? CACHE_LINE_SIZE : NATURAL_SLOT_ALIGNMENT;

   /*
    * released_sequence represents the next write generation for which this
    * slot becomes available.
    *
    * Initial slot i:
    *     released_sequence == i
    *
    * After consuming logical position p:
    *     released_sequence == p + CAPACITY
    */

   struct alignas(SLOT_ALIGNMENT) Slot
   {
      AtomicSequence released_sequence{0};

      alignas(T) std::byte storage[sizeof(T)];
   };

   /*
    * Accessed only by the single producer.
    *
    * write_index:
    *     Next logical position to construct.
    *
    * reclaim_cache:
    *     First logical position not yet known by the producer to have been
    *     consumed.
    */
   struct alignas(CACHE_LINE_SIZE) ProducerState
   {
      std::size_t write_index{0};
      std::size_t reclaim_cache{0};
   } producer_;

   /*
    * Written by the producer and read by all consumers.
    *
    * Kept away from producer_ so consumer acquire loads do not invalidate the
    * producer's private cache line.
    */
   struct alignas(CACHE_LINE_SIZE) PublishedWriteState
   {
      std::atomic<std::size_t> write_index{0};
   } published_write_;

   /*
    * Contended by consumers when reserving queue positions.
    *
    * This is separate from published_write_ so consumer CAS operations do not
    * invalidate the producer-published index cache line.
    */
   struct alignas(CACHE_LINE_SIZE) ConsumerClaimState
   {
      std::atomic<std::size_t> claim_index{0};
   } consumer_claim_;

   /*
    * Heap allocation keeps a large queue from overflowing the stack.
    *
    * C++17 aligned new guarantees the requested Slot alignment.
    */
   std::unique_ptr<Slot[]> slots_;

   [[nodiscard]]
   static constexpr std::size_t slot_index(const std::size_t logical_index) noexcept
   {
      return logical_index & MASK;
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE Slot& slot_ref(const std::size_t logical_index) noexcept
   {
      return slots_[slot_index(logical_index)];
   }

   [[nodiscard]]
   static SPMC_ALWAYS_INLINE T* slot_ptr(Slot& slot) noexcept
   {
      return std::launder(reinterpret_cast<T*>(slot.storage));
   }

   static SPMC_ALWAYS_INLINE void spin_pause() noexcept { _mm_pause(); }

   /*
    * Refresh the producer's cached consumer progress.
    *
    * Consumers can complete out of order, so the cache may only advance over
    * a contiguous run of released slots.
    *
    * The scan limit bounds the latency spike when many slots have completed.
    */
   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool refresh_reclaim_cache() noexcept
   {
      const std::size_t write_index = producer_.write_index;

      std::size_t reclaim_index = producer_.reclaim_cache;
      std::size_t scanned       = 0;

      while (reclaim_index != write_index && scanned < RECLAIM_SCAN_LIMIT)
      {
         const std::size_t expected = reclaim_index + CAPACITY;

         const std::size_t observed = slot_ref(reclaim_index).released_sequence.load(std::memory_order_acquire);

         if (observed != expected)
         {
            break;
         }

         ++reclaim_index;
         ++scanned;
      }

      producer_.reclaim_cache = reclaim_index;

      return (write_index - reclaim_index) < CAPACITY;
   }

   /*
    * Usually this is only two non-atomic integer operations.
    *
    * The producer touches consumer-written atomics only when its cached view
    * indicates that the ring may be full.
    */
   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool producer_has_space() noexcept
   {
      const std::size_t cached_size = producer_.write_index - producer_.reclaim_cache;

      if (cached_size < CAPACITY) [[likely]]
      {
         return true;
      }

      return refresh_reclaim_cache();
   }

   template<typename... Args>
   SPMC_ALWAYS_INLINE void construct_and_publish(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
   {
      const std::size_t position = producer_.write_index;

      Slot& slot = slot_ref(position);

      std::construct_at(slot_ptr(slot), std::forward<Args>(args)...);

      const std::size_t next_position = position + 1;

      producer_.write_index = next_position;

      /*
       * Publishes the constructed object.
       *
       * A consumer's acquire load of published_write_ makes the object's
       * construction visible.
       */
      published_write_.write_index.store(next_position, std::memory_order_release);
   }

public:
   /*
    * One token should be created per consumer thread.
    *
    * write_cache avoids loading published_write_ on every dequeue.
    * Because this token is consumer-local, the cache itself is non-atomic.
    */
   class alignas(CACHE_LINE_SIZE) ConsumerToken
   {
      friend class LockFreeQueueSpmcSeq;

      const LockFreeQueueSpmcSeq* owner_{nullptr};
      std::size_t                 write_cache_{0};

      explicit ConsumerToken(const LockFreeQueueSpmcSeq* owner) noexcept
         : owner_(owner)
      {}

   public:
      ConsumerToken(const ConsumerToken&)            = delete;
      ConsumerToken& operator=(const ConsumerToken&) = delete;

      ConsumerToken(ConsumerToken&&) noexcept            = default;
      ConsumerToken& operator=(ConsumerToken&&) noexcept = default;
   };

   LockFreeQueueSpmcSeq()
      : slots_(std::make_unique<Slot[]>(CAPACITY))
   {
      for (std::size_t i = 0; i < CAPACITY; ++i)
      {
         slots_[i].released_sequence.store(i, std::memory_order_relaxed);
      }
   }

   ~LockFreeQueueSpmcSeq() noexcept
   {
      /*
       * All producer and consumer threads must be stopped before queue
       * destruction.
       *
       * Destroy objects that were published but never consumed.
       */
      if constexpr (!std::is_trivially_destructible_v<T>)
      {
         const std::size_t write_index = producer_.write_index;

         for (std::size_t position = producer_.reclaim_cache; position != write_index; ++position)
         {
            Slot& slot = slot_ref(position);

            const std::size_t consumed_sequence = position + CAPACITY;

            const std::size_t observed = slot.released_sequence.load(std::memory_order_relaxed);

            if (observed != consumed_sequence)
            {
               std::destroy_at(slot_ptr(slot));
            }
         }
      }
   }

   LockFreeQueueSpmcSeq(const LockFreeQueueSpmcSeq&) = delete;

   LockFreeQueueSpmcSeq& operator=(const LockFreeQueueSpmcSeq&) = delete;

   LockFreeQueueSpmcSeq(LockFreeQueueSpmcSeq&&) = delete;

   LockFreeQueueSpmcSeq& operator=(LockFreeQueueSpmcSeq&&) = delete;

   [[nodiscard]]
   ConsumerToken make_consumer_token() const noexcept
   {
      return ConsumerToken(this);
   }

   template<typename... Args>
   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool try_emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
   {
      static_assert(std::is_nothrow_constructible_v<T, Args&&...>, "T must be nothrow constructible from Args...");

      if (!producer_has_space()) [[unlikely]]
      {
         return false;
      }

      construct_and_publish(std::forward<Args>(args)...);

      return true;
   }

   template<typename... Args>
   SPMC_ALWAYS_INLINE void emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
   {
      static_assert(std::is_nothrow_constructible_v<T, Args&&...>, "T must be nothrow constructible from Args...");

      while (!producer_has_space()) [[unlikely]]
      {
         spin_pause();
      }

      construct_and_publish(std::forward<Args>(args)...);
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool try_push(const T& value) noexcept
   {
      static_assert(std::is_nothrow_copy_constructible_v<T>, "T must be nothrow copy constructible");

      return try_emplace(value);
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool try_push(T&& value) noexcept
   {
      static_assert(std::is_nothrow_move_constructible_v<T>, "T must be nothrow move constructible");

      return try_emplace(std::move(value));
   }

   SPMC_ALWAYS_INLINE void push(const T& value) noexcept
   {
      static_assert(std::is_nothrow_copy_constructible_v<T>, "T must be nothrow copy constructible");

      emplace(value);
   }

   SPMC_ALWAYS_INLINE void push(T&& value) noexcept
   {
      static_assert(std::is_nothrow_move_constructible_v<T>, "T must be nothrow move constructible");

      emplace(std::move(value));
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool try_pop(ConsumerToken& token, T& output) noexcept
   {
      assert(token.owner_ == this && "ConsumerToken belongs to another queue");

      for (;;)
      {
         std::size_t position = consumer_claim_.claim_index.load(std::memory_order_relaxed);

         /*
          * Only reload the producer's published position when this
          * consumer's local cache says there is no available work.
          */
         if (position >= token.write_cache_) [[unlikely]]
         {
            token.write_cache_ = published_write_.write_index.load(std::memory_order_acquire);

            if (position >= token.write_cache_) [[unlikely]]
            {
               return false;
            }
         }

         /*
          * The CAS only reserves a logical position. It does not publish or
          * consume data, so relaxed ordering is sufficient.
          */
         if (consumer_claim_.claim_index.compare_exchange_weak(position,
                                                               position + 1,
                                                               std::memory_order_relaxed,
                                                               std::memory_order_relaxed))
         {
            Slot& slot  = slot_ref(position);
            T*    value = slot_ptr(slot);

            output = std::move(*value);
            std::destroy_at(value);

            /*
             * Tell the producer this specific generation has finished.
             *
             * Release prevents the producer from reconstructing the slot
             * before the move and destruction have completed.
             */
            slot.released_sequence.store(position + CAPACITY, std::memory_order_release);

            return true;
         }

         spin_pause();
      }
   }

   SPMC_ALWAYS_INLINE void pop(ConsumerToken& token, T& output) noexcept
   {
      while (!try_pop(token, output)) [[unlikely]]
      {
         spin_pause();
      }
   }

   /*
    * These describe unclaimed work, not work currently being processed by
    * consumers.
    */
   [[nodiscard]]
   bool empty_approx() const noexcept
   {
      const std::size_t claimed = consumer_claim_.claim_index.load(std::memory_order_acquire);

      const std::size_t written = published_write_.write_index.load(std::memory_order_acquire);

      return claimed >= written;
   }

   [[nodiscard]]
   std::size_t available_approx() const noexcept
   {
      const std::size_t claimed = consumer_claim_.claim_index.load(std::memory_order_acquire);

      const std::size_t written = published_write_.write_index.load(std::memory_order_acquire);

      return written >= claimed ? written - claimed : 0;
   }

   [[nodiscard]]
   static constexpr std::size_t capacity() noexcept
   {
      return CAPACITY;
   }
};

#undef SPMC_ALWAYS_INLINE
