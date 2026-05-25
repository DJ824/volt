#pragma once
#include <functional>
#include <future>
#include <atomic>
#include <cstdint>
#include <memory>
#include <semaphore>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <new>
#include <thread>
#include "work_steal_deque.h"
#include "spsc_final.h"
#include "mpsc_seq.h"
#include "task_free_list.h"

class ThreadPool {
public:
    class TaskContext;

private:
    struct Task {
        void (*run)(Task*) noexcept;
        void (*destroy)(Task*) noexcept;
    };

    template <class F, class... Args>
    struct DetachedTask : Task {
        std::decay_t<F> func;
        std::tuple<std::decay_t<Args>...> args;

        DetachedTask(void (*destroy_fn)(Task*) noexcept, F&& f, Args&&... args)
            : Task{&run_impl, destroy_fn},
              func{std::forward<F>(f)}, args{std::forward<Args>(args)...} {
        }


        static void run_impl(Task* base) noexcept {
            auto self = static_cast<DetachedTask*>(base);
            std::apply(self->func, self->args);
        }

    };

    template <class F, class... Args>
    struct DetachedContextTask : Task {
        ThreadPool* pool;
        std::decay_t<F> func;
        std::tuple<std::decay_t<Args>...> args;

        DetachedContextTask(void (*destroy_fn)(Task*) noexcept, ThreadPool* p, F&& f, Args&&... args)
            : Task{&run_impl, destroy_fn},
              pool(p),
              func(std::forward<F>(f)),
              args(std::forward<Args>(args)...) {
        }

        static void run_impl(Task* base) noexcept {
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
    struct ReturningTask : Task {
        using R = std::invoke_result_t<F, Args...>;

        std::decay_t<F> func;
        std::tuple<std::decay_t<Args>...> args;
        std::promise<R> promise;

        ReturningTask(void (*destroy_fn)(Task*) noexcept, F&& f, Args&&... args)
            : Task{&run_impl, destroy_fn},
              func{std::forward<F>(f)}, args{std::forward<Args>(args)...} {
        }

        static void run_impl(Task* base) noexcept {
            auto* self = static_cast<ReturningTask*>(base);
            try {
                if constexpr (std::is_void_v<R>) {
                    std::apply(self->func, self->args);
                    self->promise.set_value();
                } else {
                    self->promise.set_value(std::apply(self->func, self->args));
                }
            }
            catch (...) {
                self->promise.set_exception(std::current_exception());
            }
        }


    };

    template <class F, class... Args>
    struct ReturningContextTask : Task {
        using R = std::invoke_result_t<F, TaskContext&, Args...>;
        ThreadPool* pool;
        std::decay_t<F> func;
        std::tuple<std::decay_t<Args>...> args;
        std::promise<R> promise;

        ReturningContextTask(void (*destroy_fn)(Task*) noexcept, ThreadPool* p, F&& f, Args&&... args)
            : Task{&run_impl, destroy_fn},
              pool(p),
              func(std::forward<F>(f)),
              args(std::forward<Args>(args)...) {
        }

        static void run_impl(Task* base) noexcept {
            auto self = static_cast<ReturningContextTask*>(base);

            try {
                TaskContext ctx{self->pool};

                if constexpr (std::is_void_v<R>) {
                    std::apply([&](auto&&... unpacked) {
                        self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...);
                    }, self->args);
                    self->promise.set_value();
                } else
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
    static void heap_destroy(Task* base) noexcept {
        delete static_cast<T*>(base);
    }

    template <class T>
    static void pooled_destroy(Task* base) noexcept {
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
                    return new (storage) T(&pooled_destroy<T>, std::forward<Args>(args)...);
                }
                catch (...) {
                    free_list->release(storage);
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
        return std::pair<Task*, std::future<R>>{task, std::move(future)};
    }

    template <class F, class... Args>
    auto make_returning_spawning_task(TaskFreeList* free_list, F&& f, Args&&... args) {
        using T = ReturningContextTask<F, Args...>;
        using R = typename T::R;

        auto* task = make_task<T>(free_list, this, std::forward<F>(f), std::forward<Args>(args)...);
        std::future<R> future = task->promise.get_future();
        return std::pair<Task*, std::future<R>>{task, std::move(future)};
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


    static constexpr size_t kLocalQueueSize = 1024;
    static constexpr size_t kInboxQueueSize = 1024;

    struct Worker {
        using Inbox = LockFreeQueueMpscSeq<Task*, kInboxQueueSize>;
        using LocalDeque = WorkStealDeque<Task*, kLocalQueueSize>;

        ThreadPool* pool{nullptr};
        uint64_t id{0};
        Inbox inbox;
        LocalDeque local;
        std::counting_semaphore<> signal{0};
        std::thread thread;
        TaskFreeList free_list;

        Worker() {
            free_list.reserve(1024);
        }

        Worker(const Worker&) = delete;
        Worker(Worker&&) = delete;
        Worker& operator=(const Worker&) = delete;
        Worker& operator=(Worker&&) = delete;
    };


    inline static thread_local Worker* curr_worker_ = nullptr;
    bool started_{false};
    uint64_t thread_count_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<size_t> submit_cursor_{0};
    std::atomic<size_t> pending_tasks_{0};

    void run_task(Task* task) noexcept {
        task->run(task);
        task->destroy(task);

        const auto previous = pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);

        if (previous == 1 && stopping_.load(std::memory_order_acquire)) {
            wake_all_workers();
        }
    }

    void wake_all_workers() noexcept {
        for (auto& worker : workers_) {
            worker->signal.release();
        }
    }

public:
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
                worker_loop(*worker_ptr);
            });
        }
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        auto worker_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
        auto* worker = workers_[worker_id].get();
        auto [task, future] = make_returning_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
        pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
        enqueue_external(worker_id, task);
        return std::move(future);
    }

    template <class F, class... Args>
    auto submit_detached(F&& f, Args&&... args) {
        auto worker_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
        auto* worker = workers_[worker_id].get();
        Task* task = make_detached_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
        pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
        enqueue_external(worker_id, task);
    }

    template <class F, class... Args>
    auto submit_detached_ctx(F&& f, Args&&... args) {
        auto worker_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
        auto* worker = workers_[worker_id].get();
        Task* task = make_detached_spawning_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
        pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
        enqueue_external(worker_id, task);
    }

    template <class F, class... Args>
    auto spawn_detached(F&& f, Args&&... args) {
        if (curr_worker_ && curr_worker_->pool == this) {
            Task* task = make_detached_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_relaxed);
            curr_worker_->local.push_bottom(task);
            wake_all_workers();
        } else {
            Task* task = make_detached_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_relaxed);
            enqueue_external(task);
        }
    }

    template <class F, class... Args>
    auto spawn_detached_ctx(F&& f, Args&&... args) {
        if (curr_worker_ && curr_worker_->pool == this) {
            Task* task = make_detached_spawning_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_relaxed);
            curr_worker_->local.push_bottom(task);
            wake_all_workers();
        } else {
            Task* task = make_detached_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_relaxed);
            enqueue_external(task);
        }
    }

    template <class F, class... Args>
    auto spawn_returning(F&& f, Args&&... args) {
        if (curr_worker_ && curr_worker_->pool == this) {
            auto [task, future] = make_returning_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            curr_worker_->local.push_bottom(task);
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
            auto [task, future] = make_returning_spawning_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
            pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
            curr_worker_->local.push_bottom(task);
            wake_all_workers();
            return std::move(future);
        }

        auto [task, future] = make_returning_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
        pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
        enqueue_external(task);
        return std::move(future);
    }

    void wait_for_tasks() noexcept {
        while (pending_tasks_.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }


    void enqueue_external(std::size_t thread_id, Task* task) {
        auto worker = workers_[thread_id].get();
        worker->inbox.push(task);
        worker->signal.release();
    }

    void enqueue_external(Task* task) {
        auto thread_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;
        enqueue_external(thread_id, task);
    }

    void worker_loop(Worker& worker) {
        curr_worker_ = &worker;

        while (!stopping_.load(std::memory_order_acquire) || pending_tasks_.load(std::memory_order_acquire) != 0) {
            Task* task = nullptr;

            if (worker.local.try_pop_bottom(task)) {
                run_task(task);
                continue;
            }

            if (worker.inbox.try_pop(task)) {
                run_task(task);
                continue;
            }

            if (try_steal(worker, task)) {
                run_task(task);
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
        }

        curr_worker_ = nullptr;
    }


    bool try_steal(Worker& self, Task*& out) {
        for (auto& worker : workers_) {
            if (worker.get() == &self) {
                continue;
            }

            if (worker->local.try_steal_top(out)) {
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
    };
};
