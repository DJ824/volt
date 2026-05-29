#ifndef VOLT_THREAD_POOL_H
#include "thread_pool.h"
#else
#ifndef VOLT_PARALLEL_FOR_IMPL_H
#define VOLT_PARALLEL_FOR_IMPL_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <type_traits>
#include <utility>

namespace volt {

    static inline size_t ceil_div(size_t a, size_t b) noexcept {
        return (a + b - 1) / b;
    }
    template <class F>
    struct alignas(128) ThreadPool::ParallelForTask : StackTask {
        ThreadPool* pool{};
        F* func;
        size_t first{};
        size_t last{};
        size_t chunk{};
        alignas(128) std::atomic<size_t> next_block{0};
        std::atomic<size_t> pending_tasks{0};
    };

    template <class F>
    void ThreadPool::run_parallel_for_task(StackTask* base) noexcept {
        auto* task = static_cast<ParallelForTask<F>*>(base);

        while (true) {
            auto block = task->next_block.fetch_add(1);
            auto begin = task->first + block * task->chunk;
            auto end = std::min(begin + task->chunk, task->last);

            if (begin >= task->last) {
                break;
            }

            for (size_t i = begin; i < end; i++) {
                (*task->func)(i);
            }
        }
    }

    template <class F>
    void ThreadPool::finish_parallel_for_task(StackTask* base) noexcept {
        auto* task = static_cast<ParallelForTask<F>*>(base);
        task->pending_tasks.fetch_sub(1, std::memory_order_release);
        task->pool->pending_tasks_.fetch_sub(1, std::memory_order_release);
    }

    template <class F>
    void ThreadPool::parallel_for(size_t first, size_t last, F&& fn) {
        using Fn = std::decay_t<F>;
        Fn func(std::forward<F>(fn));

        ParallelForTask<Fn> task;
        task.func = &func;
        task.first = first;
        task.last = last;
        task.next_block.store(0);

        size_t participants = workers_.size() + 1;
        task.chunk = ceil_div(last - first, participants);
        task.pool = this;
        task.run = &run_parallel_for_task<Fn>;
        task.finish = &finish_parallel_for_task<Fn>;
        task.pending_tasks.store(participants);

        uint64_t gen = stack_gen_.fetch_add(2, std::memory_order_relaxed) + 2;

        for (auto& worker : workers_) {
            worker->stack_task = &task;
            worker->gen.store(gen, std::memory_order_release);
            worker->signal.release();
        }

        run_stack_task(&task);
        uint64_t done = gen | done_bit;

        for (auto& worker : workers_) {
            while (worker->gen.load(std::memory_order_relaxed) != done) {
                std::this_thread::yield();
            }
        }

    }
}

#endif
#endif
