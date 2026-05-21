#pragma once
#include <functional>
#include <future>

#include "work_steal_deque.h"
#include "spsc_final.h"
#include "mpsc_seq.h"
#include <thread>

class ThreadPool {
    struct Task {
        void (*run)(Task*) noexcept;
        void (*destroy)(Task*) noexcept;
    };

    template <class F, class... Args>
    struct DetachedTask : Task {
        std::decay_t<F> func;
        std::tuple<std::decay_t<Args>...> args;

        DetachedTask(F&& f, Args&&... args)
            : Task{&run_impl, &destroy_impl},
              func{std::forward<F>(f)}, args{std::forward<Args>(args)...} {
        }


        static void run_impl(Task* base) noexcept {
            auto self = static_cast<DetachedTask*>(base);
            std::apply(self->func, self->args);
        }

        static void destroy_impl(Task* base) noexcept {
            delete static_cast<DetachedTask*>(base);
        }
    };

    template <class F, class... Args>
    struct ReturningTask : Task {
        using R = std::invoke_result_t<F, Args...>;

        std::decay_t<F> func;
        std::tuple<std::decay_t<Args>...> args;
        std::promise<R> promise;

        ReturningTask(F&& f, Args&&... args)
            : Task{&run_impl, &destroy_impl},
              func{std::forward<F>(f)}, args{std::forward<Args>(args)...} {
        }

        static void run_impl(Task* base) noexcept {
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

        static void destroy_impl(Task* base) noexcept {
            delete static_cast<ReturningTask*>(base);
        }
    };

    template <class F, class... Args>
    auto make_returning_task(F&& f, Args&&... args) {
        using T = ReturningTask<F, Args...>;
        using R = T::R;

        auto* task = new T(std::forward<F>(f), std::forward<Args>(args)...);

        std::future<R> future = task->promise.get_future();
        return std::pair<Task*, std::future<R>>{task, std::move(future)};
    }

    template <class F, class... Args>
    auto make_detached_task(F&& f, Args&&... args) {
        using T = DetachedTask<F, Args...>;

        auto* task = new T(std::forward<F>(f), std::forward<Args>(args)...);
        return task;
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
        std::thread thread;

        Worker() = default;
        Worker(const Worker&) = delete;
        Worker(Worker&&) = delete;
        Worker& operator=(const Worker&) = delete;
        Worker& operator=(Worker&&) = delete;
    };

    uint64_t thread_count_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::counting_semaphore<> available_work_{0};
    std::atomic<bool> stopping_{false};
    std::atomic<size_t> submit_cursor_{0};
    std::atomic<size_t> pending_tasks_{0};

    void run_task(Task* task) noexcept {
        task->run(task);
        task->destroy(task);
        pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
    }

public:
    explicit ThreadPool(size_t thread_ct)
        : workers_{}, available_work_{0}, stopping_{false}, submit_cursor_{0}, pending_tasks_{0} {
        workers_.reserve(thread_ct);

        for (int i = 0; i < thread_ct; i++) {
            auto worker = std::make_unique<Worker>();
            worker->pool = this;
            worker->id = i;
            workers_.push_back(std::move(worker));
        }

        thread_count_ = thread_ct;
    }

    void start() {
        for (auto& worker : workers_) {
            auto worker_ptr = worker.get();
            worker->thread = std::thread([this, worker_ptr] {
                worker_loop(*worker_ptr);
            });
        }
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        auto [task, future] = make_returning_task(std::forward<F>(f), std::forward<Args>(args)...);

        enqueue_external(task);
        return future;
    }

    template <class F, class... Args>
    auto submit_detached(F&& f, Args&&... args) {
        Task* task = make_detached_task(std::forward<F>(f), std::forward<Args>(args)...);
        enqueue_external(task);
    }

    void enqueue_external(Task* task) {
        pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
        auto thread_id = submit_cursor_.fetch_add(1, std::memory_order_relaxed) % thread_count_;

        auto worker = workers_[thread_id].get();
        worker->inbox.push(task);
        available_work_.release();
    }

    void worker_loop(Worker& worker) {
        while (!stopping_.load(std::memory_order_acquire) && pending_tasks_.load(std::memory_order_acquire) != 0) {
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

            available_work_.acquire();
        }
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
};
