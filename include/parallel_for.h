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
    if (last <= first) {
        return;
    }

    using Fn = std::decay_t<F>;

    Fn func(std::forward<F>(fn));

    const size_t workers = workers_.size();
    if (workers == 0) [[unlikely]] {
        for (size_t i = first; i < last; ++i) {
            func(i);
        }
        return;
    }

    const bool producer_is_worker = curr_worker_ && curr_worker_->pool == this;
    const size_t enqueued_workers = producer_is_worker ? workers - 1 : workers;
    const size_t participants = enqueued_workers + 1;
    const size_t n = last - first;

    ParallelForTask<Fn> task;
    task.func = &func;
    task.first = first;
    task.last = last;
    task.chunk = std::max<size_t>(1, (n + participants - 1) / participants);
    task.run = &run_parallel_for_task<Fn>;
    task.finish = &finish_parallel_for_task<Fn>;
    task.pool = this;
    task.pending_tasks.store(participants, std::memory_order_relaxed);

    pending_tasks_.fetch_add(participants, std::memory_order_release);

    for (size_t i = 0; i < workers; i++) {
        if (producer_is_worker && workers_[i].get() == curr_worker_) {
            continue;
        }

        enqueue_stack_task(i, &task);
    }

    run_stack_task(&task);

    while (task.pending_tasks.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
    }
}

}

#endif
#endif
