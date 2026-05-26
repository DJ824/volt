#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T, size_t SIZE>
class LockFreeQueueMpmcSeq {
    using AtomicSeq = std::atomic<size_t>;

    static_assert((SIZE & (SIZE - 1)) == 0, "size must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>,
                  "LockFreeQueueMpmcSeq requires trivially copyable T");
    static_assert(std::is_trivially_default_constructible_v<T>,
                  "LockFreeQueueMpmcSeq requires trivially default-constructible T");

    static constexpr size_t CAPACITY = SIZE;
    static constexpr size_t MASK = CAPACITY - 1;

    struct alignas(CACHE_LINE_SIZE) Slot {
        T val;
        std::atomic<size_t> seq{0};
    };

    static constexpr size_t SLOT_PADDING = (CACHE_LINE_SIZE - 1) / sizeof(Slot) + 1;
    static constexpr size_t SLOT_NSLOTS = CAPACITY + 2 * SLOT_PADDING;

    struct alignas(CACHE_LINE_SIZE) ProducerState {
        std::atomic<size_t> head_{0};
    } producer_;

    struct alignas(CACHE_LINE_SIZE) ConsumerState {
        std::atomic<size_t> tail_{0};
    } consumer_;

    alignas(CACHE_LINE_SIZE) std::array<Slot, SLOT_NSLOTS> slots_{};

    [[nodiscard]] static constexpr size_t slot_index(size_t index) noexcept {
        return (index & MASK) + SLOT_PADDING;
    }

    [[nodiscard]] Slot& slot_ref(size_t index) noexcept {
        return slots_[slot_index(index)];
    }

    static void spin_pause() noexcept {
        _mm_pause();
    }

    static void wait_for_sequence(AtomicSeq& seq, size_t expected) noexcept {
        for (;;) {
            while (seq.load(std::memory_order_relaxed) != expected) {
                spin_pause();
            }

            if (seq.load(std::memory_order_acquire) == expected) {
                break;
            }
        }
    }

    template <typename U>
    void push_direct(U&& value, size_t pos) noexcept {
        Slot& slot = slot_ref(pos);
        wait_for_sequence(slot.seq, pos);
        slot.val = std::forward<U>(value);
        slot.seq.store(pos + 1, std::memory_order_release);
    }

    void pop_direct(size_t pos, T& out) noexcept {
        Slot& slot = slot_ref(pos);
        wait_for_sequence(slot.seq, pos + 1);
        out = slot.val;
        slot.seq.store(pos + CAPACITY, std::memory_order_release);
    }

public:
    explicit LockFreeQueueMpmcSeq() noexcept {
        for (size_t i = 0; i < CAPACITY; ++i) {
            slot_ref(i).seq.store(i, std::memory_order_relaxed);
        }
    }

    LockFreeQueueMpmcSeq(const LockFreeQueueMpmcSeq&) = delete;
    LockFreeQueueMpmcSeq& operator=(const LockFreeQueueMpmcSeq&) = delete;

    void emplace(const T& value) noexcept {
        push(value);
    }

    void pop(T& out) noexcept {
        const size_t pos = consumer_.tail_.fetch_add(1, std::memory_order_relaxed);
        pop_direct(pos, out);
    }

    void push(const T& value) noexcept {
        const size_t pos = producer_.head_.fetch_add(1, std::memory_order_relaxed);
        push_direct(value, pos);
    }

    [[nodiscard]] bool try_emplace(const T& value) noexcept {
        for (;;) {
            size_t pos = producer_.head_.load(std::memory_order_relaxed);
            Slot& slot = slot_ref(pos);
            const size_t observed = slot.seq.load(std::memory_order_acquire);
            const std::intptr_t diff =
                static_cast<std::intptr_t>(observed) - static_cast<std::intptr_t>(pos);

            if (diff == 0) {
                if (producer_.head_.compare_exchange_weak(pos, pos + 1,
                                                          std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
                    slot.val = value;
                    slot.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                spin_pause();
            }
        }
    }

    [[nodiscard]] bool try_push(const T& value) noexcept {
        return try_emplace(value);
    }

    [[nodiscard]] bool try_pop(T& out) noexcept {
        for (;;) {
            size_t pos = consumer_.tail_.load(std::memory_order_relaxed);
            Slot& slot = slot_ref(pos);
            const size_t observed = slot.seq.load(std::memory_order_acquire);
            const std::intptr_t diff =
                static_cast<std::intptr_t>(observed) - static_cast<std::intptr_t>(pos + 1);

            if (diff == 0) {
                if (consumer_.tail_.compare_exchange_weak(pos, pos + 1,
                                                          std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
                    out = slot.val;
                    slot.seq.store(pos + CAPACITY, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                spin_pause();
            }
        }
    }

    [[nodiscard]] std::optional<T> dequeue() noexcept {
        T value{};
        if (!try_pop(value)) {
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] bool try_dequeue(T* out) noexcept {
        T value{};
        if (!try_pop(value)) {
            return false;
        }

        if (out) {
            *out = value;
        }
        return true;
    }

    bool enqueue(const T& item) noexcept {
        return try_emplace(item);
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] size_t size() const noexcept {
        const size_t tail = consumer_.tail_.load(std::memory_order_acquire);
        const size_t head = producer_.head_.load(std::memory_order_acquire);
        return head >= tail ? head - tail : 0;
    }

    [[nodiscard]] constexpr size_t capacity() const noexcept {
        return CAPACITY;
    }
};
