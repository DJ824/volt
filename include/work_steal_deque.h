#pragma once

#include <array>
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T, std::size_t N>
class WorkStealDeque {
    static_assert((N & (N - 1)) == 0, "capacity must be a power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "WorkStealDeque requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "WorkStealDeque requires trivially default-constructible T");

    static constexpr std::size_t CAPACITY = N;
    static constexpr std::size_t MASK = CAPACITY - 1;
    static constexpr std::size_t PADDING = (CACHE_LINE_SIZE - 1) / sizeof(T) + 1;
    static constexpr std::size_t NSLOTS = CAPACITY + 2 * PADDING;

    struct alignas(CACHE_LINE_SIZE) OwnerState {
        std::size_t bottom_{0};
        std::size_t top_cache_{0};
    } owner_;

    struct alignas(CACHE_LINE_SIZE) SharedTop {
        std::atomic<std::size_t> top_{0};
    } shared_top_;

    struct alignas(CACHE_LINE_SIZE) PublishedBottom {
        std::atomic<std::size_t> bottom_{0};
    } published_bottom_;

    alignas(CACHE_LINE_SIZE) std::array<T, NSLOTS> buffer_{};

    [[nodiscard]] static constexpr std::size_t slot_index(std::size_t index) noexcept {
        return (index & MASK) + PADDING;
    }

    [[nodiscard]] T& slot_ref(std::size_t index) noexcept {
        return buffer_[slot_index(index)];
    }

    [[nodiscard]] const T& slot_ref(std::size_t index) const noexcept {
        return buffer_[slot_index(index)];
    }

    static void spin_pause() noexcept {
        _mm_pause();
    }

public:
    using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

    WorkStealDeque() noexcept = default;

    WorkStealDeque(const WorkStealDeque&) = delete;
    WorkStealDeque& operator=(const WorkStealDeque&) = delete;

    template <typename U>
    [[nodiscard]] bool try_push_bottom(U&& value) noexcept {
        const std::size_t bottom = owner_.bottom_;
        std::size_t top_cache = owner_.top_cache_;

        if (bottom - top_cache >= CAPACITY) [[unlikely]] {
            top_cache = shared_top_.top_.load(std::memory_order_acquire);
            owner_.top_cache_ = top_cache;
            if (bottom - top_cache >= CAPACITY) [[unlikely]] {
                return false;
            }
        }

        slot_ref(bottom) = std::forward<U>(value);
        owner_.bottom_ = bottom + 1;
        published_bottom_.bottom_.store(bottom + 1, std::memory_order_release);
        return true;
    }

    template <typename O>
    void push_bottom(O&& value) noexcept {
        while (!try_push_bottom(value)) {
            spin_pause();
        }
    }

    template <typename U>
    [[nodiscard]] std::size_t try_bulk_push_bottom(U first, std::size_t count) noexcept {
        if (count == 0) [[unlikely]] {
            return 0;
        }

        const std::size_t bottom = owner_.bottom_;
        std::size_t top_cache = owner_.top_cache_;
        std::size_t used = bottom - top_cache;

        if (used >= CAPACITY) [[unlikely]] {
            owner_.top_cache_ = shared_top_.top_.load(std::memory_order_acquire);
            used = bottom - owner_.top_cache_;
            if (used >= CAPACITY) [[unlikely]] {
                return 0;
            }
        }

        const std::size_t available = CAPACITY - used;
        const std::size_t pushed = std::min(count, available);
        for (std::size_t i = 0; i < pushed; ++i) {
            slot_ref(bottom + i) = first[i];
        }

        const std::size_t next_bottom = bottom + pushed;
        owner_.bottom_ = next_bottom;
        published_bottom_.bottom_.store(next_bottom, std::memory_order_release);
        return pushed;
    }

    template <typename I>
    void bulk_push_bottom(I first, std::size_t count) noexcept {
        std::size_t offset = 0;
        while (offset < count) {
            const std::size_t pushed = try_bulk_push_bottom(first + offset, count - offset);
            if (pushed == 0) {
                spin_pause();
                continue;
            }
            offset += pushed;
        }
    }

    [[nodiscard]] bool try_pop_bottom(T& out) noexcept {
        std::size_t bottom = owner_.bottom_;
        std::size_t top_cache = owner_.top_cache_;

        if (bottom == top_cache) [[unlikely]] {
            top_cache = shared_top_.top_.load(std::memory_order_acquire);
            owner_.top_cache_ = top_cache;
            if (bottom == top_cache) [[unlikely]] {
                return false;
            }
        }


        bottom -= 1;
        owner_.bottom_ = bottom;
        published_bottom_.bottom_.store(bottom, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        size_t top = shared_top_.top_.load(std::memory_order_relaxed);
        if (top > bottom) [[unlikely]] {
            const std::size_t next = bottom + 1;
            owner_.bottom_ = next;
            owner_.top_cache_ = top;
            published_bottom_.bottom_.store(next, std::memory_order_relaxed);
            return false;
        }

        if (top == bottom) [[unlikely]] {
            size_t expected = top;
            if (!shared_top_.top_.compare_exchange_strong(expected, top + 1,
                                                          std::memory_order_seq_cst,
                                                          std::memory_order_relaxed)) {
                owner_.bottom_ = expected;
                owner_.top_cache_ = expected;
                published_bottom_.bottom_.store(expected, std::memory_order_relaxed);
                return false;
            }

            owner_.bottom_ = bottom + 1;
            owner_.top_cache_ = bottom + 1;
            published_bottom_.bottom_.store(bottom + 1, std::memory_order_relaxed);
            out = slot_ref(bottom);
            return true;
        }

        owner_.top_cache_ = top;
        out = slot_ref(bottom);
        return true;
    }

    [[nodiscard]] value_type pop_bottom() noexcept {
        T value{};
        if (!try_pop_bottom(value)) {
            if constexpr (std::is_pointer_v<T>) {
                return T{nullptr};
            } else {
                return std::nullopt;
            }
        }
        return value;
    }

    [[nodiscard]] bool try_steal_top(T& out) noexcept {
        std::size_t top = shared_top_.top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const std::size_t bottom = published_bottom_.bottom_.load(std::memory_order_acquire);
        if (top >= bottom) [[unlikely]] {
            return false;
        }

        const T value = slot_ref(top);
        if (!shared_top_.top_.compare_exchange_strong(top, top + 1,
                                                      std::memory_order_seq_cst,
                                                      std::memory_order_relaxed)) {
            return false;
        }

        out = value;
        return true;
    }

    [[nodiscard]] value_type steal_top() noexcept {
        T value{};
        if (!try_steal_top(value)) {
            if constexpr (std::is_pointer_v<T>) {
                return T{nullptr};
            } else {
                return std::nullopt;
            }
        }
        return value;
    }

    template <typename O>
    [[nodiscard]] bool try_push(O&& value) noexcept {
        return try_push_bottom(std::forward<O>(value));
    }

    template <typename I>
    [[nodiscard]] std::size_t try_bulk_push(I first, std::size_t count) noexcept {
        return try_bulk_push_bottom(first, count);
    }

    [[nodiscard]] value_type pop() noexcept {
        return pop_bottom();
    }

    [[nodiscard]] value_type steal() noexcept {
        return steal_top();
    }

    [[nodiscard]] value_type steal_with_feedback(std::size_t& num_empty_steals) noexcept {
        T value{};
        if (try_steal_top(value)) {
            num_empty_steals = 0;
            return value;
        }

        ++num_empty_steals;
        if constexpr (std::is_pointer_v<T>) {
            return T{nullptr};
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t top = shared_top_.top_.load(std::memory_order_acquire);
        const std::size_t bottom = published_bottom_.bottom_.load(std::memory_order_acquire);
        return bottom >= top ? bottom - top : 0;
    }

    [[nodiscard]] constexpr std::size_t capacity() const noexcept {
        return CAPACITY;
    }
};
