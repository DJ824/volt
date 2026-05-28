#pragma once
#define VOLT_THREAD_POOL_H

#include <chrono>
#include <functional>
#include <future>
#include <atomic>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <new>
#include <thread>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#include "workers.h"

namespace volt {
    class ThreadPool {
    public:
        class TaskContext;

    private:
        using HeapTask = details::HeapTask;
        using StackTask = details::StackTask;
        using Worker = details::Worker;

        template <class F>
        struct ParallelForTask;

        template <class F>
        static void run_parallel_for_task(StackTask* base) noexcept;

        template <class F>
        static void finish_parallel_for_task(StackTask* base) noexcept;


        template <class F, class... Args>
        struct DetachedTask : HeapTask {
            std::decay_t<F> func;
            std::tuple<std::decay_t<Args>...> args;

            DetachedTask(void (*destroy_fn)(HeapTask*) noexcept, F&& f, Args&&... args)
                : HeapTask{&run_impl, destroy_fn},
                  func{std::forward<F>(f)}, args{std::forward<Args>(args)...} {
            }


            static void run_impl(HeapTask* base) noexcept {
                auto self = static_cast<DetachedTask*>(base);
                std::apply(self->func, self->args);
            }
        };

        template <class F, class... Args>
        struct DetachedContextTask : HeapTask {
            ThreadPool* pool;
            std::decay_t<F> func;
            std::tuple<std::decay_t<Args>...> args;

            DetachedContextTask(void (*destroy_fn)(HeapTask*) noexcept, ThreadPool* p, F&& f, Args&&... args)
                : HeapTask{&run_impl, destroy_fn},
                  pool(p),
                  func(std::forward<F>(f)),
                  args(std::forward<Args>(args)...) {
            }

            static void run_impl(HeapTask* base) noexcept {
                auto self = static_cast<DetachedContextTask*>(base);

                try {
                    TaskContext ctx{self->pool};
                    std::apply([&](auto&&... unpacked) {
                        self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...);
                    }, self->args);
                }
                catch (...) {
                    std::terminate();
                }
            }
        };

        template <class F, class... Args>
        struct ReturningTask : HeapTask {
            using R = std::invoke_result_t<F, Args...>;

            std::decay_t<F> func;
            std::tuple<std::decay_t<Args>...> args;
            std::promise<R> promise;

            ReturningTask(void (*destroy_fn)(HeapTask*) noexcept, F&& f, Args&&... args)
                : HeapTask{&run_impl, destroy_fn},
                  func{std::forward<F>(f)}, args{std::forward<Args>(args)...} {
            }

            static void run_impl(HeapTask* base) noexcept {
                auto* self = static_cast<ReturningTask*>(base);
                try {
                    if constexpr (std::is_void_v<R>) {
                        std::apply(self->func, self->args);
                        self->promise.set_value();
                    }
                    else {
                        self->promise.set_value(std::apply(self->func, self->args));
                    }
                }
                catch (...) {
                    self->promise.set_exception(std::current_exception());
                }
            }
        };

        template <class F, class... Args>
        struct ReturningContextTask : HeapTask {
            using R = std::invoke_result_t<F, TaskContext&, Args...>;
            ThreadPool* pool;
            std::decay_t<F> func;
            std::tuple<std::decay_t<Args>...> args;
            std::promise<R> promise;

            ReturningContextTask(void (*destroy_fn)(HeapTask*) noexcept, ThreadPool* p, F&& f, Args&&... args)
                : HeapTask{&run_impl, destroy_fn},
                  pool(p),
                  func(std::forward<F>(f)),
                  args(std::forward<Args>(args)...) {
            }

            static void run_impl(HeapTask* base) noexcept {
                auto self = static_cast<ReturningContextTask*>(base);

                try {
                    TaskContext ctx{self->pool};

                    if constexpr (std::is_void_v<R>) {
                        std::apply([&](auto&&... unpacked) {
                            self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...);
                        }, self->args);
                        self->promise.set_value();
                    }
                    else
                        self->promise.set_value(std::apply([&](auto&&... unpacked) {
                            return self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...);
                        }, self->args));
                }
                catch (...) {
                    self->promise.set_exception(std::current_exception());
                }
            }
        };

        template <class T>
        static void heap_destroy(HeapTask* base) noexcept {
            delete static_cast<T*>(base);
        }

        template <class T>
        static void pooled_destroy(HeapTask* base) noexcept {
            auto* self = static_cast<T*>(base);
            void* storage = self;
            self->~T();
            curr_worker_->free_list.release(storage);
        }

        template <class T, class... Args>
        T* make_task(TaskFreeList* free_list, Args&&... args) {
            if constexpr (sizeof(T) <= TaskFreeList::block_size() &&
                alignof(T) <= TaskFreeList::block_align()) {
                if (free_list != nullptr) {
                    void* storage = free_list->acquire();
                    try {
                        return new(storage) T(&pooled_destroy<T>, std::forward<Args>(args)...);
                    }
                    catch (...) {
                        free_list->release(storage);
                        // log err
                        throw;
                    }
                }
            }

            return new T(&heap_destroy<T>, std::forward<Args>(args)...);
        }

        template <class F, class... Args>
        auto make_returning_task(TaskFreeList* free_list, F&& f, Args&&... args) {
            using T = ReturningTask<F, Args...>;
            using R = typename T::R;

            auto* task = make_task<T>(free_list, std::forward<F>(f), std::forward<Args>(args)...);

            std::future<R> future = task->promise.get_future();
            return std::pair<HeapTask*, std::future<R>>{task, std::move(future)};
        }

        template <class F, class... Args>
        auto make_returning_spawning_task(TaskFreeList* free_list, F&& f, Args&&... args) {
            using T = ReturningContextTask<F, Args...>;
            using R = typename T::R;

            auto* task = make_task<T>(free_list, this, std::forward<F>(f), std::forward<Args>(args)...);
            std::future<R> future = task->promise.get_future();
            return std::pair<HeapTask*, std::future<R>>{task, std::move(future)};
        }

        template <class F, class... Args>
        auto make_detached_task(TaskFreeList* free_list, F&& f, Args&&... args) {
            using T = DetachedTask<F, Args...>;

            return make_task<T>(free_list, std::forward<F>(f), std::forward<Args>(args)...);
        }

        template <class F, class... Args>
        auto make_detached_spawning_task(TaskFreeList* free_list, F&& f, Args&&... args) {
            using T = DetachedContextTask<F, Args...>;

            return make_task<T>(free_list, this, std::forward<F>(f), std::forward<Args>(args)...);
        }


        inline static thread_local Worker* curr_worker_ = nullptr;
        bool started_{false};
        uint64_t thread_count_;
        std::vector<std::unique_ptr<Worker>> workers_;
        std::atomic<bool> stopping_{false};
        std::atomic<size_t> submit_cursor_{0};
        std::atomic<size_t> pending_tasks_{0};

        void run_heap_task(HeapTask* task) noexcept {
            task->run(task);
            task->destroy(task);

            const auto previous = pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);

            if (previous == 1 && stopping_.load(std::memory_order_acquire)) {
                wake_all_workers();
            }
        }

        void run_stack_task(StackTask* task) noexcept {
            task->run(task);
            task->finish(task);
        }

        void wake_all_workers() noexcept {
            for (auto& worker : workers_) {
                // worker->wake_word.fetch_add(1, std::memory_order_release);
                // futex_wake(worker->wake_word, 1);
                worker->signal.release();
            }
        }

        [[nodiscard]] static uint64_t default_thread_count() noexcept {
#ifdef __linux__
            cpu_set_t allowed;
            CPU_ZERO(&allowed);

            if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
                uint64_t count = 0;
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                    if (CPU_ISSET(cpu, &allowed)) {
                        ++count;
                    }
                }

                if (count != 0) {
                    return count;
                }
            }
#endif

            const auto count = std::thread::hardware_concurrency();
            return count == 0 ? 1 : count;
        }

        static void pin_current_thread(uint64_t worker_id) noexcept {
#ifdef __linux__
            cpu_set_t allowed;
            CPU_ZERO(&allowed);

            if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
                return;
            }

            uint64_t allowed_count = 0;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &allowed)) {
                    ++allowed_count;
                }
            }

            if (allowed_count == 0) {
                return;
            }

            const uint64_t target_index = worker_id % allowed_count;
            uint64_t seen = 0;
            int target_cpu = -1;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (!CPU_ISSET(cpu, &allowed)) {
                    continue;
                }

                if (seen == target_index) {
                    target_cpu = cpu;
                    break;
                }

                ++seen;
            }

            if (target_cpu < 0) {
                return;
            }

            cpu_set_t target;
            CPU_ZERO(&target);
            CPU_SET(target_cpu, &target);
            static_cast<void>(pthread_setaffinity_np(pthread_self(), sizeof(target), &target));
#else
            static_cast<void>(worker_id);
#endif
        }

        void enqueue_stack_task(size_t worker_idx, StackTask* task) noexcept {
            auto worker = workers_[worker_idx].get();

            worker->stack_inbox.enqueue(task);
            worker->signal.release();
        }

    public:
        ThreadPool() : ThreadPool(default_thread_count()) {
        }

        explicit ThreadPool(size_t thread_ct) {
            workers_.reserve(thread_ct);

            for (size_t i = 0; i < thread_ct; i++) {
                auto worker = std::make_unique<Worker>();
                worker->pool = this;
                worker->id = i;
                workers_.push_back(std::move(worker));
            }

            thread_count_ = thread_ct;
        }

        ~ThreadPool() {
            stop();
            join();
        }

        void start() {
            if (started_) {
                return;
            }

            started_ = true;

            for (auto& worker : workers_) {
                auto worker_ptr = worker.get();
                worker->thread = std::thread([this, worker_ptr] {
                    // pin_current_thread(worker_ptr->id);
                    worker_loop(*worker_ptr);
                });
            }
        }

        template <class F>
        void parallel_for(size_t first, size_t last, F&& fn);

        template <class F, class... Args>
        auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
            auto worker_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
            auto* worker = workers_[worker_id].get();
            auto [task, future] = make_returning_task(&worker->free_list, std::forward<F>(f),
                                                      std::forward<Args>(args)...);
            // auto [task, future] = make_returning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            enqueue_external(worker_id, task);
            return std::move(future);
        }

        template <class F, class... Args>
        auto submit_detached(F&& f, Args&&... args) {
            auto worker_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
            auto* worker = workers_[worker_id].get();
            HeapTask* task = make_detached_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
            // HeapTask* task = make_detached_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            enqueue_external(worker_id, task);
        }

        template <class F, class... Args>
        auto submit_detached_ctx(F&& f, Args&&... args) {
            auto worker_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
            auto* worker = workers_[worker_id].get();
            HeapTask* task = make_detached_spawning_task(&worker->free_list, std::forward<F>(f),
                                                     std::forward<Args>(args)...);
            // HeapTask* task = make_detached_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);

            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            enqueue_external(worker_id, task);
        }

        template <class F, class... Args>
        auto spawn_detached(F&& f, Args&&... args) {
            if (curr_worker_ && curr_worker_->pool == this) {
                HeapTask* task = make_detached_task(&curr_worker_->free_list, std::forward<F>(f),
                                                std::forward<Args>(args)...);
                pending_tasks_.fetch_add(1, std::memory_order_relaxed);
                curr_worker_->deque.push_bottom(task);
                wake_all_workers();
            }
            else {
                HeapTask* task = make_detached_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
                pending_tasks_.fetch_add(1, std::memory_order_relaxed);
                enqueue_external(task);
            }
        }

        template <class F, class... Args>
        auto spawn_detached_ctx(F&& f, Args&&... args) {
            if (curr_worker_ && curr_worker_->pool == this) {
                HeapTask* task = make_detached_spawning_task(&curr_worker_->free_list, std::forward<F>(f),
                                                         std::forward<Args>(args)...);
                pending_tasks_.fetch_add(1, std::memory_order_relaxed);
                curr_worker_->deque.push_bottom(task);
                wake_all_workers();
            }
            else {
                HeapTask* task = make_detached_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
                pending_tasks_.fetch_add(1, std::memory_order_relaxed);
                enqueue_external(task);
            }
        }

        template <class F, class... Args>
        auto spawn_returning(F&& f, Args&&... args) {
            if (curr_worker_ && curr_worker_->pool == this) {
                auto [task, future] = make_returning_task(&curr_worker_->free_list, std::forward<F>(f),
                                                          std::forward<Args>(args)...);
                pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
                curr_worker_->deque.push_bottom(task);
                wake_all_workers();
                return std::move(future);
            }

            auto [task, future] = make_returning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            enqueue_external(task);
            return std::move(future);
        }

        template <class F, class... Args>
        auto spawn_returning_ctx(F&& f, Args&&... args) {
            if (curr_worker_ && curr_worker_->pool == this) {
                auto [task, future] = make_returning_spawning_task(&curr_worker_->free_list, std::forward<F>(f),
                                                                   std::forward<Args>(args)...);
                pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
                curr_worker_->deque.push_bottom(task);
                wake_all_workers();
                return std::move(future);
            }

            auto [task, future] =
                make_returning_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            enqueue_external(task);
            return std::move(future);
        }

        void wait_for_tasks() noexcept {
            while (pending_tasks_.load(std::memory_order_acquire) != 0) {
                std::this_thread::yield();
            }
        }


        void enqueue_external(std::size_t thread_id, HeapTask* task) {
            auto worker = workers_[thread_id].get();
            worker->inbox.push(task);
            worker->signal.release();
            // worker->wake_word.fetch_add(1, std::memory_order_release);
            // futex_wake(worker->wake_word, 1);
        }

        void enqueue_external(HeapTask* task) {
            auto thread_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
            enqueue_external(thread_id, task);
        }

        void worker_loop(Worker& worker) {
            curr_worker_ = &worker;

            while (!stopping_.load(std::memory_order_acquire) || pending_tasks_.load(std::memory_order_acquire) != 0) {
                HeapTask* task = nullptr;
                StackTask* stack_task = nullptr;

                if (worker.deque.try_pop_bottom(task)) {
                    run_heap_task(task);
                    continue;
                }

                if (worker.inbox.try_pop(task)) {
                    run_heap_task(task);
                    continue;
                }

                if (try_steal_deque(worker, task)) {
                    run_heap_task(task);
                    continue;
                }

                if (try_steal_local(worker, task)) {
                    run_heap_task(task);
                    continue;
                }

                if (worker.stack_inbox.try_pop(stack_task)) {
                    run_stack_task(stack_task);
                    continue;
                }

                if (worker.stack_deque.try_pop_bottom(stack_task)) {
                    run_stack_task(stack_task);
                    continue;
                }

                if (stopping_.load(std::memory_order_acquire) &&
                    pending_tasks_.load(std::memory_order_acquire) == 0) {
                    break;
                }

                if (pending_tasks_.load(std::memory_order_acquire) != 0) {
                    std::this_thread::yield();
                    continue;
                }

                worker.signal.acquire();

                // auto token = worker.wake_word.load(std::memory_order_relaxed);


                // if (worker.local.try_pop_bottom(task) ||
                //     worker.inbox.try_pop(task) ||
                //     try_steal_deque(worker, task) ||
                //     try_steal_local(worker, task)) {
                //     run_task(task);
                //     continue;
                // }
                //
                // if (pending_tasks_.load(std::memory_order_acquire) == 0) {
                //     futex_wait(worker.wake_word, token);
                // }

                // if (stopping_.load(std::memory_order_acquire) && pending_tasks_.load(std::memory_order_acquire) == 0) {
                //     break;
                // }
            }

            curr_worker_ = nullptr;
        }

        bool try_run_once(Worker& worker) {
            HeapTask* task = nullptr;

            if (worker.deque.try_pop_bottom(task)) {
                run_heap_task(task);
                return true;
            }

            if (worker.inbox.try_pop(task)) {
                run_heap_task(task);
                return true;
            }


            if (try_steal_deque(worker, task)) {
                run_heap_task(task);
                return true;
            }

            return false;
        }

        bool try_steal_local(Worker& self, HeapTask*& out) {
            for (auto& worker : workers_) {
                if (worker.get() == &self) {
                    continue;
                }

                if (worker->inbox.try_pop(out)) {
                    return true;
                }
            }
            return false;
        }


        bool try_steal_deque(Worker& self, HeapTask*& out) {
            for (auto& worker : workers_) {
                if (worker.get() == &self) {
                    continue;
                }

                if (worker->deque.try_steal_top(out)) {
                    return true;
                }
            }

            return false;
        }

        void stop() {
            stopping_.store(true, std::memory_order_release);
            wake_all_workers();
        }

        void join() {
            for (auto& worker : workers_) {
                if (worker->thread.joinable()) {
                    worker->thread.join();
                }
            }
        }

        class TaskContext {
            ThreadPool* pool_;

        public:
            explicit TaskContext(ThreadPool* pool) : pool_{pool} {
            }

            template <class F, class... Args>
            void spawn_detached(F&& f, Args&&... args) {
                pool_->spawn_detached(std::forward<F>(f), std::forward<Args>(args)...);
            }

            template <class F, class... Args>
            void spawn_detached_ctx(F&& f, Args&&... args) {
                pool_->spawn_detached_ctx(std::forward<F>(f),
                                          std::forward<Args>(args)...);
            }

            template <class F, class... Args>
            auto spawn_returning(F&& f, Args&&... args) {
                return pool_->spawn_returning(
                    std::forward<F>(f),
                    std::forward<Args>(args)...
                );
            }

            template <class F, class... Args>
            auto spawn_returning_ctx(F&& f, Args&&... args) {
                return pool_->spawn_returning_ctx(
                    std::forward<F>(f),
                    std::forward<Args>(args)...
                );
            }

            template <class R>
            R get(std::future<R>& fut) {
                while (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    if (!pool_->try_run_once(*curr_worker_)) {
                        std::this_thread::yield();
                    }
                }

                if constexpr (std::is_void_v<R>) {
                    fut.get();
                }
                else {
                    return fut.get();
                }
            }
        };
    };
}

#include "parallel_for.h"
