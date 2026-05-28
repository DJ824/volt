#pragma once

#include <cstddef>
#include <cstdint>
#include <semaphore>
#include <thread>

#include "mpmc_seq.h"
#include "task_free_list.h"
#include "work_steal_deque.h"

namespace volt {
    class ThreadPool;

    namespace details {
        inline constexpr std::size_t kLocalQueueSize = 8192;
        inline constexpr std::size_t kInboxQueueSize = 8192;

        struct HeapTask {
            void (*run)(HeapTask*) noexcept;
            void (*destroy)(HeapTask*) noexcept;
        };

        struct StackTask {
            void (*run)(StackTask*) noexcept;
            void (*finish)(StackTask*) noexcept;
        };

        struct Worker {
            // using Inbox = LockFreeQueueMpscSeq<HeapTask*, kInboxQueueSize>;
            using Inbox = LockFreeQueueMpmcSeq<HeapTask*, kInboxQueueSize>;
            using StackInbox = LockFreeQueueMpmcSeq<StackTask*, kLocalQueueSize>;
            using LocalDeque = WorkStealDeque<HeapTask*, kLocalQueueSize>;
            using StackDeque = WorkStealDeque<StackTask*, kLocalQueueSize>;

            ThreadPool* pool{nullptr};
            uint64_t id{0};
            Inbox inbox;
            LocalDeque deque;
            StackInbox stack_inbox;
            StackDeque stack_deque;
            std::binary_semaphore signal{0};
            // std::atomic<uint32_t> wake_word{0};
            std::thread thread;
            TaskFreeList free_list;

            Worker() {
                free_list.reserve(8192);
            }
        };
    }
}
