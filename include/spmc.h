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
#define CACHE_LINE_SIZE 128
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SPMC_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SPMC_ALWAYS_INLINE __forceinline
#else
#define SPMC_ALWAYS_INLINE inline
#endif

template<typename T, std::size_t size, bool padSlots = false, std::size_t reclaimScanLimit = 32>
class LockFreeQueueSpmcSeq
{
   static_assert(size > 0, "size must be non-zero");
   static_assert((size & (size - 1)) == 0, "size must be a power of two");
   static_assert((CACHE_LINE_SIZE & (CACHE_LINE_SIZE - 1)) == 0, "CACHE_LINE_SIZE must be a power of two");
   static_assert(reclaimScanLimit > 0, "reclaimScanLimit must be non-zero");
   static_assert(std::atomic<std::size_t>::is_always_lock_free, "size_t atomics must be lock-free");
   static_assert(std::is_nothrow_move_assignable_v<T>, "T must be nothrow move assignable for pop(T&)");
   static_assert(std::is_nothrow_destructible_v<T>, "T must be nothrow destructible");

   static constexpr std::size_t kCapacity = size;
   static constexpr std::size_t kMask     = kCapacity - 1;

   using AtomicSequence = std::atomic<std::size_t>;

   static constexpr std::size_t kNaturalSlotAlignment = alignof(T) > alignof(AtomicSequence) ? alignof(T) :
                                                                                               alignof(AtomicSequence);

   static constexpr std::size_t kSlotAlignment =
      padSlots && CACHE_LINE_SIZE > kNaturalSlotAlignment ? CACHE_LINE_SIZE : kNaturalSlotAlignment;

   /*
    * releasedSequence represents the next write generation for which this
    * slot becomes available.
    *
    * Initial slot i:
    *     releasedSequence == i
    *
    * After consuming logical position p:
    *     releasedSequence == p + kCapacity
    */

   struct alignas(kSlotAlignment) Slot
   {
      AtomicSequence releasedSequence{0};

      alignas(T) std::byte storage[sizeof(T)];
   };

   /*
    * Accessed only by the single producer.
    *
    * writeIndex:
    *     Next logical position to construct.
    *
    * reclaimCache:
    *     First logical position not yet known by the producer to have been
    *     consumed.
    */
   struct alignas(CACHE_LINE_SIZE) ProducerState
   {
      std::size_t writeIndex{0};
      std::size_t reclaimCache{0};
   } producer_;

   /*
    * Written by the producer and read by all consumers.
    *
    * Kept away from producer_ so consumer acquire loads do not invalidate the
    * producer's private cache line.
    */
   struct alignas(CACHE_LINE_SIZE) PublishedWriteState
   {
      std::atomic<std::size_t> writeIndex{0};
   } publishedWrite_;

   /*
    * Contended by consumers when reserving queue positions.
    *
    * This is separate from publishedWrite_ so consumer CAS operations do not
    * invalidate the producer-published index cache line.
    */
   struct alignas(CACHE_LINE_SIZE) ConsumerClaimState
   {
      std::atomic<std::size_t> claimIndex{0};
   } consumerClaim_;

   /*
    * Heap allocation keeps a large queue from overflowing the stack.
    *
    * C++17 aligned new guarantees the requested Slot alignment.
    */
   std::unique_ptr<Slot[]> slots_;

   [[nodiscard]]
   static constexpr std::size_t slotIndex(const std::size_t logicalIndex) noexcept
   {
      return logicalIndex & kMask;
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE Slot& slotRef(const std::size_t logicalIndex) noexcept
   {
      return slots_[slotIndex(logicalIndex)];
   }

   [[nodiscard]]
   static SPMC_ALWAYS_INLINE T* slotPtr(Slot& slot) noexcept
   {
      return std::launder(reinterpret_cast<T*>(slot.storage));
   }

   static SPMC_ALWAYS_INLINE void spinPause() noexcept { _mm_pause(); }

   /*
    * Refresh the producer's cached consumer progress.
    *
    * Consumers can complete out of order, so the cache may only advance over
    * a contiguous run of released slots.
    *
    * The scan limit bounds the latency spike when many slots have completed.
    */
   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool refreshReclaimCache() noexcept
   {
      const std::size_t writeIndex = producer_.writeIndex;

      std::size_t reclaimIndex = producer_.reclaimCache;
      std::size_t scanned      = 0;

      while (reclaimIndex != writeIndex && scanned < reclaimScanLimit)
      {
         const std::size_t expected = reclaimIndex + kCapacity;

         const std::size_t observed = slotRef(reclaimIndex).releasedSequence.load(std::memory_order_acquire);

         if (observed != expected)
         {
            break;
         }

         ++reclaimIndex;
         ++scanned;
      }

      producer_.reclaimCache = reclaimIndex;

      return (writeIndex - reclaimIndex) < kCapacity;
   }

   /*
    * Usually this is only two non-atomic integer operations.
    *
    * The producer touches consumer-written atomics only when its cached view
    * indicates that the ring may be full.
    */
   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool producerHasSpace() noexcept
   {
      const std::size_t cachedSize = producer_.writeIndex - producer_.reclaimCache;

      if (cachedSize < kCapacity) [[likely]]
      {
         return true;
      }

      return refreshReclaimCache();
   }

   template<typename... Args>
   SPMC_ALWAYS_INLINE void constructAndPublish(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
   {
      const std::size_t position = producer_.writeIndex;

      Slot& slot = slotRef(position);

      std::construct_at(slotPtr(slot), std::forward<Args>(args)...);

      const std::size_t nextPosition = position + 1;

      producer_.writeIndex = nextPosition;

      /*
       * Publishes the constructed object.
       *
       * A consumer's acquire load of publishedWrite_ makes the object's
       * construction visible.
       */
      publishedWrite_.writeIndex.store(nextPosition, std::memory_order_release);
   }

public:
   /*
    * One token should be created per consumer thread.
    *
    * write_cache avoids loading publishedWrite_ on every dequeue.
    * Because this token is consumer-local, the cache itself is non-atomic.
    */
   class alignas(CACHE_LINE_SIZE) ConsumerToken
   {
      friend class LockFreeQueueSpmcSeq;

      const LockFreeQueueSpmcSeq* owner_{nullptr};
      std::size_t                 writeCache_{0};

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
      : slots_(std::make_unique<Slot[]>(kCapacity))
   {
      for (std::size_t i = 0; i < kCapacity; ++i)
      {
         slots_[i].releasedSequence.store(i, std::memory_order_relaxed);
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
         const std::size_t writeIndex = producer_.writeIndex;

         for (std::size_t position = producer_.reclaimCache; position != writeIndex; ++position)
         {
            Slot& slot = slotRef(position);

            const std::size_t consumedSequence = position + kCapacity;

            const std::size_t observed = slot.releasedSequence.load(std::memory_order_relaxed);

            if (observed != consumedSequence)
            {
               std::destroy_at(slotPtr(slot));
            }
         }
      }
   }

   LockFreeQueueSpmcSeq(const LockFreeQueueSpmcSeq&) = delete;

   LockFreeQueueSpmcSeq& operator=(const LockFreeQueueSpmcSeq&) = delete;

   LockFreeQueueSpmcSeq(LockFreeQueueSpmcSeq&&) = delete;

   LockFreeQueueSpmcSeq& operator=(LockFreeQueueSpmcSeq&&) = delete;

   [[nodiscard]]
   ConsumerToken makeConsumerToken() const noexcept
   {
      return ConsumerToken(this);
   }

   template<typename... Args>
   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool tryEmplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
   {
      static_assert(std::is_nothrow_constructible_v<T, Args&&...>, "T must be nothrow constructible from Args...");

      if (!producerHasSpace()) [[unlikely]]
      {
         return false;
      }

      constructAndPublish(std::forward<Args>(args)...);

      return true;
   }

   template<typename... Args>
   SPMC_ALWAYS_INLINE void emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
   {
      static_assert(std::is_nothrow_constructible_v<T, Args&&...>, "T must be nothrow constructible from Args...");

      while (!producerHasSpace()) [[unlikely]]
      {
         spinPause();
      }

      constructAndPublish(std::forward<Args>(args)...);
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool tryPush(const T& value) noexcept
   {
      static_assert(std::is_nothrow_copy_constructible_v<T>, "T must be nothrow copy constructible");

      return tryEmplace(value);
   }

   [[nodiscard]]
   SPMC_ALWAYS_INLINE bool tryPush(T&& value) noexcept
   {
      static_assert(std::is_nothrow_move_constructible_v<T>, "T must be nothrow move constructible");

      return tryEmplace(std::move(value));
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
   SPMC_ALWAYS_INLINE bool tryPop(ConsumerToken& token, T& output) noexcept
   {
      assert(token.owner_ == this && "ConsumerToken belongs to another queue");

      for (;;)
      {
         std::size_t position = consumerClaim_.claimIndex.load(std::memory_order_relaxed);

         /*
          * Only reload the producer's published position when this
          * consumer's local cache says there is no available work.
          */
         if (position >= token.writeCache_) [[unlikely]]
         {
            token.writeCache_ = publishedWrite_.writeIndex.load(std::memory_order_acquire);

            if (position >= token.writeCache_) [[unlikely]]
            {
               return false;
            }
         }

         /*
          * The CAS only reserves a logical position. It does not publish or
          * consume data, so relaxed ordering is sufficient.
          */
         if (consumerClaim_.claimIndex.compare_exchange_weak(position,
                                                             position + 1,
                                                             std::memory_order_relaxed,
                                                             std::memory_order_relaxed))
         {
            Slot& slot  = slotRef(position);
            T*    value = slotPtr(slot);

            output = std::move(*value);
            std::destroy_at(value);

            /*
             * Tell the producer this specific generation has finished.
             *
             * Release prevents the producer from reconstructing the slot
             * before the move and destruction have completed.
             */
            slot.releasedSequence.store(position + kCapacity, std::memory_order_release);

            return true;
         }

         spinPause();
      }
   }

   SPMC_ALWAYS_INLINE void pop(ConsumerToken& token, T& output) noexcept
   {
      while (!tryPop(token, output)) [[unlikely]]
      {
         spinPause();
      }
   }

   /*
    * These describe unclaimed work, not work currently being processed by
    * consumers.
    */
   [[nodiscard]]
   bool emptyApprox() const noexcept
   {
      const std::size_t claimed = consumerClaim_.claimIndex.load(std::memory_order_acquire);

      const std::size_t written = publishedWrite_.writeIndex.load(std::memory_order_acquire);

      return claimed >= written;
   }

   [[nodiscard]]
   std::size_t availableApprox() const noexcept
   {
      const std::size_t claimed = consumerClaim_.claimIndex.load(std::memory_order_acquire);

      const std::size_t written = publishedWrite_.writeIndex.load(std::memory_order_acquire);

      return written >= claimed ? written - claimed : 0;
   }

   [[nodiscard]]
   static constexpr std::size_t capacity() noexcept
   {
      return kCapacity;
   }
};

#undef SPMC_ALWAYS_INLINE