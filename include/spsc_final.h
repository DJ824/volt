#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LOCK_FREE_QUEUE2_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define LOCK_FREE_QUEUE2_ALWAYS_INLINE inline
#endif

template <typename T, size_t SIZE,
          bool DIRECT_SLOTS =
              std::is_trivially_copyable_v<T> &&
              std::is_trivially_default_constructible_v<T> &&
              std::is_trivially_destructible_v<T>>
class LockFreeQueue2;

template <typename T, size_t SIZE>
class LockFreeQueue2<T, SIZE, true> {
    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "requires trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t NSLOTS = CAPACITY + 2 * PADDING;

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        size_t write_index_{0};
        size_t read_index_cache_{0};
        const size_t padding_cache_{PADDING};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) PublishedWriteIndex {
        std::atomic<size_t> write_index_{0};
    } published_write_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        size_t read_index_{0};
        size_t write_index_cache_{0};
        const size_t capacity_cache_{CAPACITY};
    } consumer_;

    struct alignas(CACHE_LINE_SIZE) PublishedReadIndex {
        std::atomic<size_t> read_index_{0};
    } published_read_;

    alignas(CACHE_LINE_SIZE) std::array<T, NSLOTS> buffer_;

    [[nodiscard]] static constexpr size_t slot_index(size_t index) noexcept {
        return (index & MASK) + PADDING;
    }

    static LOCK_FREE_QUEUE2_ALWAYS_INLINE void multi_pause() noexcept {
        _mm_pause();
        _mm_pause();
    }

public:
    explicit LockFreeQueue2() = default;

    LockFreeQueue2(const LockFreeQueue2&) = delete;
    LockFreeQueue2& operator=(const LockFreeQueue2&) = delete;

    LOCK_FREE_QUEUE2_ALWAYS_INLINE void emplace(const T& value) noexcept {
        const size_t write_index = producer_.write_index_;

        while (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
            producer_.read_index_cache_ = published_read_.read_index_.load(std::memory_order_acquire);
            if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
                multi_pause();
            }
        }

        buffer_[slot_index(write_index)] = value;
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index,
                                            std::memory_order_release);
    }

    LOCK_FREE_QUEUE2_ALWAYS_INLINE void emplace(T&& value) noexcept {
        const size_t write_index = producer_.write_index_;

        while (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
            producer_.read_index_cache_ = published_read_.read_index_.load(std::memory_order_acquire);
            if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
                multi_pause();
            }
        }

        buffer_[slot_index(write_index)] = std::move(value);
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index,
                                            std::memory_order_release);
    }

    template <typename... Args>
    LOCK_FREE_QUEUE2_ALWAYS_INLINE void emplace(Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>) {
        static_assert(std::is_constructible_v<T, Args&&...>,
                      "[emplace] requires constructible T");
        const size_t write_index = producer_.write_index_;

        while (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
            producer_.read_index_cache_ =
                published_read_.read_index_.load(std::memory_order_acquire);
            if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
                multi_pause();
            }
        }

        buffer_[slot_index(write_index)] = T(std::forward<Args>(args)...);
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index,
                                            std::memory_order_release);
    }

    LOCK_FREE_QUEUE2_ALWAYS_INLINE void pop(T& out) noexcept {
        const size_t read_index = consumer_.read_index_;

        while (read_index == consumer_.write_index_cache_) [[unlikely]] {
            consumer_.write_index_cache_ =
                published_write_.write_index_.load(std::memory_order_acquire);
            if (read_index == consumer_.write_index_cache_) [[unlikely]] {
                multi_pause();
            }
        }

        out = buffer_[slot_index(read_index)];
        const size_t next_read_index = read_index + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index,
                                          std::memory_order_release);
    }

    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE bool try_emplace(
        const T& value) noexcept {
        const size_t write_index = producer_.write_index_;
        size_t read_index_cache = producer_.read_index_cache_;

        if (write_index == read_index_cache + MASK) [[unlikely]] {
            read_index_cache =
                published_read_.read_index_.load(std::memory_order_acquire);
            producer_.read_index_cache_ = read_index_cache;
            if (write_index == read_index_cache + MASK) [[unlikely]] {
                return false;
            }
        }

        buffer_[slot_index(write_index)] = value;
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index,
                                            std::memory_order_release);
        return true;
    }

    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE bool try_emplace(
        T&& value) noexcept {
        const size_t write_index = producer_.write_index_;
        size_t read_index_cache = producer_.read_index_cache_;

        if (write_index == read_index_cache + MASK) [[unlikely]] {
            read_index_cache =
                published_read_.read_index_.load(std::memory_order_acquire);
            producer_.read_index_cache_ = read_index_cache;
            if (write_index == read_index_cache + MASK) [[unlikely]] {
                return false;
            }
        }

        buffer_[slot_index(write_index)] = std::move(value);
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index,
                                            std::memory_order_release);
        return true;
    }

    template <typename... Args>
    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE bool try_emplace(
        Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        static_assert(std::is_constructible_v<T, Args&&...>,
                      "requires constructible T");
        const size_t write_index = producer_.write_index_;
        size_t read_index_cache = producer_.read_index_cache_;

        if (write_index == read_index_cache + MASK) [[unlikely]] {
            read_index_cache =
                published_read_.read_index_.load(std::memory_order_acquire);
            producer_.read_index_cache_ = read_index_cache;
            if (write_index == read_index_cache + MASK) [[unlikely]] {
                return false;
            }
        }

        buffer_[slot_index(write_index)] = T(std::forward<Args>(args)...);
        const size_t next_write_index = write_index + 1;
        producer_.write_index_ = next_write_index;
        published_write_.write_index_.store(next_write_index,
                                            std::memory_order_release);
        return true;
    }

    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE bool try_pop(T& out) noexcept {
        const size_t read_index = consumer_.read_index_;
        size_t write_index_cache = consumer_.write_index_cache_;

        if (read_index == write_index_cache) [[unlikely]] {
            write_index_cache =
                published_write_.write_index_.load(std::memory_order_acquire);
            consumer_.write_index_cache_ = write_index_cache;
            if (read_index == write_index_cache) [[unlikely]] {
                return false;
            }
        }

        out = buffer_[slot_index(read_index)];
        const size_t next_read_index = read_index + 1;
        consumer_.read_index_ = next_read_index;
        published_read_.read_index_.store(next_read_index,
                                          std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return published_write_.write_index_.load(std::memory_order_acquire) ==
            published_read_.read_index_.load(std::memory_order_acquire);
    }

    size_t size() const noexcept {
        const size_t read_index =
            published_read_.read_index_.load(std::memory_order_acquire);
        const size_t write_index =
            published_write_.write_index_.load(std::memory_order_acquire);
        return write_index - read_index;
    }

    size_t capacity() const noexcept {
        return CAPACITY - 1;
    }
};

template <typename T, size_t SIZE>
class LockFreeQueue2<T, SIZE, false> {
    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_move_constructible_v<T>,
                  "requires move-constructible T");
    static_assert(std::is_destructible_v<T>,
                  "requires destructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr size_t NSLOTS = CAPACITY + 2 * PADDING;

    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        size_t write_index_{0};
        size_t read_index_cache_{0};
        const size_t padding_cache_{PADDING};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) PublishedWriteIndex {
        std::atomic<size_t> write_index_{0};
    } published_write_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        size_t read_index_{0};
        size_t write_index_cache_{0};
        const size_t capacity_cache_{CAPACITY};
    } consumer_;

    struct alignas(CACHE_LINE_SIZE) PublishedReadIndex {
        std::atomic<size_t> read_index_{0};
    } published_read_;

    alignas(CACHE_LINE_SIZE) std::array<Storage, NSLOTS> buffer_;

    [[nodiscard]] static constexpr size_t slot_index(size_t index) noexcept {
        return (index & MASK) + PADDING;
    }

    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE T* slot_ptr(size_t index) noexcept {
        return std::launder(
            reinterpret_cast<T*>(std::addressof(buffer_[index])));
    }

    template <typename... Args>
    LOCK_FREE_QUEUE2_ALWAYS_INLINE void construct_slot(
        size_t index,
        Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        std::construct_at(slot_ptr(index), std::forward<Args>(args)...);
    }

    LOCK_FREE_QUEUE2_ALWAYS_INLINE void destroy_slot(size_t index) noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_at(slot_ptr(index));
        }
    }

    static LOCK_FREE_QUEUE2_ALWAYS_INLINE void multi_pause() noexcept {
        _mm_pause();
        _mm_pause();
    }

public:
    explicit LockFreeQueue2() = default;

    ~LockFreeQueue2() {
        size_t read_index =
            published_read_.read_index_.load(std::memory_order_relaxed);
        const size_t write_index =
            published_write_.write_index_.load(std::memory_order_relaxed);
        while (read_index != write_index) {
            destroy_slot(slot_index(read_index));
            ++read_index;
        }
    }

    LockFreeQueue2(const LockFreeQueue2&) = delete;
    LockFreeQueue2& operator=(const LockFreeQueue2&) = delete;

    template <typename... Args>
    LOCK_FREE_QUEUE2_ALWAYS_INLINE void emplace(Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>) {
        static_assert(std::is_constructible_v<T, Args&&...>,
                      "[emplace] requires constructible T");
        const size_t write_index = producer_.write_index_;

        while (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
            producer_.read_index_cache_ =
                published_read_.read_index_.load(std::memory_order_acquire);
            if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
                multi_pause();
            }
        }

        const size_t slot = slot_index(write_index);
        construct_slot(slot, std::forward<Args>(args)...);
        producer_.write_index_ = write_index + 1;
        published_write_.write_index_.store(write_index + 1,
                                            std::memory_order_release);
    }

    LOCK_FREE_QUEUE2_ALWAYS_INLINE void pop(T& out) noexcept {
        static_assert(std::is_move_assignable_v<T>,
                      "[pop] requires move-assignable T");
        const size_t read_index = consumer_.read_index_;

        while (read_index == consumer_.write_index_cache_) [[unlikely]] {
            consumer_.write_index_cache_ =
                published_write_.write_index_.load(std::memory_order_acquire);
            if (read_index == consumer_.write_index_cache_) [[unlikely]] {
                multi_pause();
            }
        }

        const size_t slot = slot_index(read_index);
        out = std::move(*slot_ptr(slot));
        destroy_slot(slot);
        consumer_.read_index_ = read_index + 1;
        published_read_.read_index_.store(read_index + 1,
                                          std::memory_order_release);
    }

    template <typename... Args>
    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE bool try_emplace(
        Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        static_assert(std::is_constructible_v<T, Args&&...>,
                      "requires constructible T");
        const size_t write_index = producer_.write_index_;

        if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
            producer_.read_index_cache_ =
                published_read_.read_index_.load(std::memory_order_acquire);
            if (write_index == producer_.read_index_cache_ + MASK) [[unlikely]] {
                return false;
            }
        }

        const size_t slot = slot_index(write_index);
        construct_slot(slot, std::forward<Args>(args)...);
        producer_.write_index_ = write_index + 1;
        published_write_.write_index_.store(write_index + 1,
                                            std::memory_order_release);
        return true;
    }

    [[nodiscard]] LOCK_FREE_QUEUE2_ALWAYS_INLINE bool try_pop(T& out) noexcept {
        static_assert(std::is_move_assignable_v<T>,
                      "requires move-assignable T");
        const size_t read_index = consumer_.read_index_;

        if (read_index == consumer_.write_index_cache_) [[unlikely]] {
            consumer_.write_index_cache_ =
                published_write_.write_index_.load(std::memory_order_acquire);
            if (read_index == consumer_.write_index_cache_) [[unlikely]] {
                return false;
            }
        }

        const size_t slot = slot_index(read_index);
        out = std::move(*slot_ptr(slot));
        destroy_slot(slot);
        consumer_.read_index_ = read_index + 1;
        published_read_.read_index_.store(read_index + 1,
                                          std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return published_write_.write_index_.load(std::memory_order_acquire) ==
            published_read_.read_index_.load(std::memory_order_acquire);
    }

    size_t size() const noexcept {
        const size_t read_index =
            published_read_.read_index_.load(std::memory_order_acquire);
        const size_t write_index =
            published_write_.write_index_.load(std::memory_order_acquire);
        return write_index - read_index;
    }

    size_t capacity() const noexcept {
        return CAPACITY - 1;
    }
};

#undef LOCK_FREE_QUEUE2_ALWAYS_INLINE
