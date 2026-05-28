#pragma once
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

inline int futex_wait(std::atomic<uint32_t>& word, uint32_t expected) {
    auto* addr = reinterpret_cast<uint32_t*>(&word);

    return static_cast<int>(syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0));
}

inline int futex_wake(std::atomic<uint32_t>& word, uint32_t count) {
    auto* addr = reinterpret_cast <uint32_t*>(&word);

    return static_cast<int>(syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, count, nullptr, nullptr, 0));
}

