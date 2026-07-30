#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 128
#endif

template<typename T, std::size_t N>
class WorkStealDeque
{
   static_assert((N & (N - 1)) == 0, "capacity must be a power of 2");
   static_assert(std::is_trivially_copyable_v<T>, "WorkStealDeque requires trivially copyable T");
   static_assert(std::is_trivially_default_constructible_v<T>, "WorkStealDeque requires trivially default-constructible T");

   static constexpr std::size_t kCapacity  = N;
   static constexpr std::size_t kMask      = kCapacity - 1;
   static constexpr std::size_t kPadding   = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
   static constexpr std::size_t kSlotCount = kCapacity + 2 * kPadding;

   struct alignas(CACHE_LINE_SIZE) OwnerState
   {
      std::size_t bottom_{0};
      std::size_t topCache_{0};
   } owner_;

   struct alignas(CACHE_LINE_SIZE) SharedTop
   {
      std::atomic<std::size_t> top_{0};
   } sharedTop_;

   struct alignas(CACHE_LINE_SIZE) PublishedBottom
   {
      std::atomic<std::size_t> bottom_{0};
   } publishedBottom_;

   alignas(CACHE_LINE_SIZE) std::array<T, kSlotCount> buffer_{};

   [[nodiscard]]
   static constexpr std::size_t slotIndex(std::size_t index) noexcept
   {
      return (index & kMask) + kPadding;
   }

   [[nodiscard]]
   T& slotRef(std::size_t index) noexcept
   {
      return buffer_[slotIndex(index)];
   }

   [[nodiscard]]
   const T& slotRef(std::size_t index) const noexcept
   {
      return buffer_[slotIndex(index)];
   }

   static void spinPause() noexcept { _mm_pause(); }

public:
   using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

   WorkStealDeque() noexcept = default;

   WorkStealDeque(const WorkStealDeque&)            = delete;
   WorkStealDeque& operator=(const WorkStealDeque&) = delete;

   template<typename U>
   [[nodiscard]]
   bool tryPushBottom(U&& value) noexcept
   {
      const std::size_t bottom   = owner_.bottom_;
      std::size_t       topCache = owner_.topCache_;

      if (bottom - topCache >= kCapacity) [[unlikely]]
      {
         topCache         = sharedTop_.top_.load(std::memory_order_acquire);
         owner_.topCache_ = topCache;
         if (bottom - topCache >= kCapacity) [[unlikely]]
         {
            return false;
         }
      }

      slotRef(bottom) = std::forward<U>(value);
      owner_.bottom_  = bottom + 1;
      publishedBottom_.bottom_.store(bottom + 1, std::memory_order_release);
      return true;
   }

   template<typename O>
   void pushBottom(O&& value) noexcept
   {
      while (!tryPushBottom(value))
      {
         spinPause();
      }
   }

   template<typename U>
   [[nodiscard]]
   std::size_t tryBulkPushBottom(U first, std::size_t count) noexcept
   {
      if (count == 0) [[unlikely]]
      {
         return 0;
      }

      const std::size_t bottom   = owner_.bottom_;
      std::size_t       topCache = owner_.topCache_;
      std::size_t       used     = bottom - topCache;

      if (used >= kCapacity) [[unlikely]]
      {
         owner_.topCache_ = sharedTop_.top_.load(std::memory_order_acquire);
         used             = bottom - owner_.topCache_;
         if (used >= kCapacity) [[unlikely]]
         {
            return 0;
         }
      }

      const std::size_t available = kCapacity - used;
      const std::size_t pushed    = std::min(count, available);
      for (std::size_t i = 0; i < pushed; ++i)
      {
         slotRef(bottom + i) = first[i];
      }

      const std::size_t nextBottom = bottom + pushed;
      owner_.bottom_               = nextBottom;
      publishedBottom_.bottom_.store(nextBottom, std::memory_order_release);
      return pushed;
   }

   template<typename I>
   void bulkPushBottom(I first, std::size_t count) noexcept
   {
      std::size_t offset = 0;
      while (offset < count)
      {
         const std::size_t pushed = tryBulkPushBottom(first + offset, count - offset);
         if (pushed == 0)
         {
            spinPause();
            continue;
         }
         offset += pushed;
      }
   }

   [[nodiscard]]
   bool tryPopBottom(T& out) noexcept
   {
      std::size_t bottom   = owner_.bottom_;
      std::size_t topCache = owner_.topCache_;

      if (bottom == topCache) [[unlikely]]
      {
         topCache         = sharedTop_.top_.load(std::memory_order_acquire);
         owner_.topCache_ = topCache;
         if (bottom == topCache) [[unlikely]]
         {
            return false;
         }
      }


      bottom -= 1;
      owner_.bottom_ = bottom;
      publishedBottom_.bottom_.store(bottom, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_seq_cst);

      size_t top = sharedTop_.top_.load(std::memory_order_relaxed);
      if (top > bottom) [[unlikely]]
      {
         const std::size_t next = bottom + 1;
         owner_.bottom_         = next;
         owner_.topCache_       = top;
         publishedBottom_.bottom_.store(next, std::memory_order_relaxed);
         return false;
      }

      if (top == bottom) [[unlikely]]
      {
         size_t expected = top;
         if (!sharedTop_.top_.compare_exchange_strong(expected, top + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
         {
            owner_.bottom_   = expected;
            owner_.topCache_ = expected;
            publishedBottom_.bottom_.store(expected, std::memory_order_relaxed);
            return false;
         }

         owner_.bottom_   = bottom + 1;
         owner_.topCache_ = bottom + 1;
         publishedBottom_.bottom_.store(bottom + 1, std::memory_order_relaxed);
         out = slotRef(bottom);
         return true;
      }

      owner_.topCache_ = top;
      out              = slotRef(bottom);
      return true;
   }

   [[nodiscard]]
   value_type popBottom() noexcept
   {
      T value{};
      if (!tryPopBottom(value))
      {
         if constexpr (std::is_pointer_v<T>)
         {
            return T{nullptr};
         }
         else
         {
            return std::nullopt;
         }
      }
      return value;
   }

   [[nodiscard]]
   bool tryStealTop(T& out) noexcept
   {
      std::size_t top = sharedTop_.top_.load(std::memory_order_acquire);
      std::atomic_thread_fence(std::memory_order_seq_cst);
      const std::size_t bottom = publishedBottom_.bottom_.load(std::memory_order_acquire);
      if (top >= bottom) [[unlikely]]
      {
         return false;
      }

      const T value = slotRef(top);
      if (!sharedTop_.top_.compare_exchange_strong(top, top + 1, std::memory_order_seq_cst, std::memory_order_relaxed))
      {
         return false;
      }

      out = value;
      return true;
   }

   [[nodiscard]]
   value_type stealTop() noexcept
   {
      T value{};
      if (!tryStealTop(value))
      {
         if constexpr (std::is_pointer_v<T>)
         {
            return T{nullptr};
         }
         else
         {
            return std::nullopt;
         }
      }
      return value;
   }

   template<typename O>
   [[nodiscard]]
   bool tryPush(O&& value) noexcept
   {
      return tryPushBottom(std::forward<O>(value));
   }

   template<typename I>
   [[nodiscard]]
   std::size_t tryBulkPush(I first, std::size_t count) noexcept
   {
      return tryBulkPushBottom(first, count);
   }

   [[nodiscard]]
   value_type pop() noexcept
   {
      return popBottom();
   }

   [[nodiscard]]
   value_type steal() noexcept
   {
      return stealTop();
   }

   [[nodiscard]]
   value_type stealWithFeedback(std::size_t& numEmptySteals) noexcept
   {
      T value{};
      if (tryStealTop(value))
      {
         numEmptySteals = 0;
         return value;
      }

      ++numEmptySteals;
      if constexpr (std::is_pointer_v<T>)
      {
         return T{nullptr};
      }
      else
      {
         return std::nullopt;
      }
   }

   [[nodiscard]]
   bool empty() const noexcept
   {
      return size() == 0;
   }

   [[nodiscard]]
   std::size_t size() const noexcept
   {
      const std::size_t top    = sharedTop_.top_.load(std::memory_order_acquire);
      const std::size_t bottom = publishedBottom_.bottom_.load(std::memory_order_acquire);
      return bottom >= top ? bottom - top : 0;
   }

   [[nodiscard]]
   constexpr std::size_t capacity() const noexcept
   {
      return kCapacity;
   }
};