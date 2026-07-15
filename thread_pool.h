#pragma once
#define VOLT_THREAD_POOL_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <immintrin.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#include "workers.h"

namespace volt
{

class ThreadPool
{
public:
   class TaskContext;

private:
   using HeapTask   = details::HeapTask;
   using Worker     = details::Worker;
   using InboxToken = typename Worker::Inbox::ConsumerToken;

   template<class F, class... Args>
   struct DetachedTask : HeapTask
   {
      std::decay_t<F>                   func;
      std::tuple<std::decay_t<Args>...> args;

      DetachedTask(void (*destroy_fn)(HeapTask*) noexcept, F&& f, Args&&... args)
         : HeapTask{&run_impl, destroy_fn}
         , func{std::forward<F>(f)}
         , args{std::forward<Args>(args)...}
      {}


      static void run_impl(HeapTask* base) noexcept
      {
         auto self = static_cast<DetachedTask*>(base);
         std::apply(self->func, self->args);
      }
   };

   template<class F, class... Args>
   struct DetachedContextTask : HeapTask
   {
      ThreadPool*                       pool;
      std::decay_t<F>                   func;
      std::tuple<std::decay_t<Args>...> args;

      DetachedContextTask(void (*destroy_fn)(HeapTask*) noexcept, ThreadPool* p, F&& f, Args&&... args)
         : HeapTask{&run_impl, destroy_fn}
         , pool(p)
         , func(std::forward<F>(f))
         , args(std::forward<Args>(args)...)
      {}

      static void run_impl(HeapTask* base) noexcept
      {
         auto self = static_cast<DetachedContextTask*>(base);

         try
         {
            TaskContext ctx{self->pool};
            std::apply([&](auto&&... unpacked) { self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...); },
                       self->args);
         }
         catch (...)
         {
            std::terminate();
         }
      }
   };

   template<class F, class... Args>
   struct ReturningTask : HeapTask
   {
      using R = std::invoke_result_t<F, Args...>;

      std::decay_t<F>                   func;
      std::tuple<std::decay_t<Args>...> args;
      std::promise<R>                   promise;

      ReturningTask(void (*destroy_fn)(HeapTask*) noexcept, F&& f, Args&&... args)
         : HeapTask{&run_impl, destroy_fn}
         , func{std::forward<F>(f)}
         , args{std::forward<Args>(args)...}
      {}

      static void run_impl(HeapTask* base) noexcept
      {
         auto* self = static_cast<ReturningTask*>(base);
         try
         {
            if constexpr (std::is_void_v<R>)
            {
               std::apply(self->func, self->args);
               self->promise.set_value();
            }
            else
            {
               self->promise.set_value(std::apply(self->func, self->args));
            }
         }
         catch (...)
         {
            self->promise.set_exception(std::current_exception());
         }
      }
   };

   template<class F, class... Args>
   struct ReturningContextTask : HeapTask
   {
      using R = std::invoke_result_t<F, TaskContext&, Args...>;
      ThreadPool*                       pool;
      std::decay_t<F>                   func;
      std::tuple<std::decay_t<Args>...> args;
      std::promise<R>                   promise;

      ReturningContextTask(void (*destroy_fn)(HeapTask*) noexcept, ThreadPool* p, F&& f, Args&&... args)
         : HeapTask{&run_impl, destroy_fn}
         , pool(p)
         , func(std::forward<F>(f))
         , args(std::forward<Args>(args)...)
      {}

      static void run_impl(HeapTask* base) noexcept
      {
         auto self = static_cast<ReturningContextTask*>(base);

         try
         {
            TaskContext ctx{self->pool};

            if constexpr (std::is_void_v<R>)
            {
               std::apply([&](auto&&... unpacked) { self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...); },
                          self->args);
               self->promise.set_value();
            }
            else
               self->promise.set_value(
                  std::apply([&](auto&&... unpacked)
                             { return self->func(ctx, std::forward<decltype(unpacked)>(unpacked)...); },
                             self->args));
         }
         catch (...)
         {
            self->promise.set_exception(std::current_exception());
         }
      }
   };

   template<class T>
   static void heap_destroy(HeapTask* base) noexcept
   {
      delete static_cast<T*>(base);
   }

   template<class T>
   static void pooled_destroy(HeapTask* base) noexcept
   {
      auto* self    = static_cast<T*>(base);
      void* storage = self;
      self->~T();
      curr_worker_->free_list.release(storage);
   }

   template<class T, class... Args>
   T* make_task(TaskFreeList* free_list, Args&&... args)
   {
      if constexpr (sizeof(T) <= TaskFreeList::block_size() && alignof(T) <= TaskFreeList::block_align())
      {
         if (free_list != nullptr)
         {
            void* storage = free_list->acquire();
            try
            {
               return new (storage) T(&pooled_destroy<T>, std::forward<Args>(args)...);
            }
            catch (...)
            {
               free_list->release(storage);
               // log err
               throw;
            }
         }
      }

      return new T(&heap_destroy<T>, std::forward<Args>(args)...);
   }

   template<class F, class... Args>
   auto make_returning_task(TaskFreeList* free_list, F&& f, Args&&... args)
   {
      using T = ReturningTask<F, Args...>;
      using R = typename T::R;

      auto* task = make_task<T>(free_list, std::forward<F>(f), std::forward<Args>(args)...);

      std::future<R> future = task->promise.get_future();
      return std::pair<HeapTask*, std::future<R>>{task, std::move(future)};
   }

   template<class F, class... Args>
   auto make_returning_spawning_task(TaskFreeList* free_list, F&& f, Args&&... args)
   {
      using T = ReturningContextTask<F, Args...>;
      using R = typename T::R;

      auto*          task   = make_task<T>(free_list, this, std::forward<F>(f), std::forward<Args>(args)...);
      std::future<R> future = task->promise.get_future();
      return std::pair<HeapTask*, std::future<R>>{task, std::move(future)};
   }

   template<class F, class... Args>
   auto make_detached_task(TaskFreeList* free_list, F&& f, Args&&... args)
   {
      using T = DetachedTask<F, Args...>;

      return make_task<T>(free_list, std::forward<F>(f), std::forward<Args>(args)...);
   }

   template<class F, class... Args>
   auto make_detached_spawning_task(TaskFreeList* free_list, F&& f, Args&&... args)
   {
      using T = DetachedContextTask<F, Args...>;

      return make_task<T>(free_list, this, std::forward<F>(f), std::forward<Args>(args)...);
   }


   inline static thread_local Worker* curr_worker_ = nullptr;

   bool                                 started_{false};
   std::size_t                          thread_count_{0};
   std::vector<std::unique_ptr<Worker>> workers_;

   // One token per [consumer worker][source inbox].
   std::vector<std::vector<InboxToken>> inbox_tokens_;

   std::atomic<bool>        stopping_{false};
   std::size_t              submit_cursor_{0};
   std::atomic<std::size_t> pending_tasks_{0};

   void run_heap_task(HeapTask* task) noexcept
   {
      task->run(task);
      task->destroy(task);

      const auto previous = pending_tasks_.fetch_sub(1, std::memory_order_release);

      if (previous == 1 && stopping_.load(std::memory_order_acquire))
      {
         wake_all_workers();
      }
   }

   void wake_all_workers() noexcept
   {
      for (auto& worker : workers_)
      {
         worker->wakeSequence.fetch_add(1, std::memory_order_release);

         worker->wakeSequence.notify_one();
      }
   }

   [[nodiscard]]
   static uint64_t default_thread_count() noexcept
   {
#ifdef __linux__
      cpu_set_t allowed;
      CPU_ZERO(&allowed);

      if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
      {
         uint64_t count = 0;
         for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
         {
            if (CPU_ISSET(cpu, &allowed))
            {
               ++count;
            }
         }

         if (count != 0)
         {
            return count;
         }
      }
#endif

      const auto count = std::thread::hardware_concurrency();
      return count == 0 ? 1 : count;
   }

   static void pin_current_thread(uint64_t worker_id) noexcept
   {
#ifdef __linux__
      cpu_set_t allowed;
      CPU_ZERO(&allowed);

      if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
      {
         return;
      }

      uint64_t allowed_count = 0;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      {
         if (CPU_ISSET(cpu, &allowed))
         {
            ++allowed_count;
         }
      }

      if (allowed_count == 0)
      {
         return;
      }

      const uint64_t target_index = worker_id % allowed_count;
      uint64_t       seen         = 0;
      int            target_cpu   = -1;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      {
         if (!CPU_ISSET(cpu, &allowed))
         {
            continue;
         }

         if (seen == target_index)
         {
            target_cpu = cpu;
            break;
         }

         ++seen;
      }

      if (target_cpu < 0)
      {
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

   [[nodiscard]]
   bool try_pop_inbox(Worker& consumer, Worker& source, HeapTask*& task) noexcept
   {
      const auto consumer_id = static_cast<std::size_t>(consumer.id);
      const auto source_id   = static_cast<std::size_t>(source.id);

      return source.inbox.try_pop(inbox_tokens_[consumer_id][source_id], task);
   }

   void enqueue_external(std::size_t worker_id, HeapTask* task)
   {
      Worker& worker = *workers_[worker_id];

      worker.inbox.push(task);

      worker.wakeSequence.fetch_add(1, std::memory_order_release);

      worker.wakeSequence.notify_one();
   }

   void enqueue_external(HeapTask* task)
   {
      auto thread_id = submit_cursor_++ % thread_count_;
      enqueue_external(thread_id, task);
   }

public:
   ThreadPool()
      : ThreadPool(2)
   {}

   explicit ThreadPool(std::size_t thread_ct)
      : thread_count_{std::max<std::size_t>(1, thread_ct)}
   {
      workers_.reserve(thread_count_);

      for (std::size_t i = 0; i < thread_count_; ++i)
      {
         auto worker         = std::make_unique<Worker>();
         worker->pool        = this;
         worker->id          = i;
         worker->next_victim = (i + 1) % thread_count_;
         workers_.push_back(std::move(worker));
      }

      inbox_tokens_.reserve(thread_count_);

      for (std::size_t consumer = 0; consumer < thread_count_; ++consumer)
      {
         inbox_tokens_.emplace_back();
         auto& tokens = inbox_tokens_.back();
         tokens.reserve(thread_count_);

         for (std::size_t source = 0; source < thread_count_; ++source)
         {
            tokens.emplace_back(workers_[source]->inbox.make_consumer_token());
         }
      }
   }

   ~ThreadPool()
   {
      stop();
      join();
   }

   void start()
   {
      if (started_)
      {
         return;
      }

      started_ = true;

      for (auto& worker : workers_)
      {
         auto worker_ptr = worker.get();
         worker->thread  = std::thread(
            [this, worker_ptr]
            {
               // pin_current_thread(worker_ptr->id);
               worker_loop(*worker_ptr);
            });
      }
   }

   template<class F, class... Args>
   auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
   {
      auto  worker_id     = submit_cursor_++ % thread_count_;
      auto* worker        = workers_[worker_id].get();
      auto [task, future] = make_returning_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
      // auto [task, future] = make_returning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
      pending_tasks_.fetch_add(1, std::memory_order_relaxed);
      enqueue_external(worker_id, task);
      return std::move(future);
   }

   template<class F, class... Args>
   auto submit_detached(F&& f, Args&&... args)
   {
      auto  worker_id = submit_cursor_++ % thread_count_;
      auto* worker    = workers_[worker_id].get();
      //HeapTask* task      = make_detached_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
      HeapTask* task = make_detached_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
      pending_tasks_.fetch_add(1, std::memory_order_relaxed);
      enqueue_external(worker_id, task);
   }

   template<class F, class... Args>
   auto submit_detached_ctx(F&& f, Args&&... args)
   {
      auto      worker_id = submit_cursor_++ % thread_count_;
      auto*     worker    = workers_[worker_id].get();
      HeapTask* task = make_detached_spawning_task(&worker->free_list, std::forward<F>(f), std::forward<Args>(args)...);
      // HeapTask* task = make_detached_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);

      pending_tasks_.fetch_add(1, std::memory_order_seq_cst);
      enqueue_external(worker_id, task);
   }

   template<class F, class... Args>
   auto spawn_detached(F&& f, Args&&... args)
   {
      if (curr_worker_ && curr_worker_->pool == this)
      {
         HeapTask* task = make_detached_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
         pending_tasks_.fetch_add(1, std::memory_order_relaxed);
         curr_worker_->local.push_bottom(task);
         wake_all_workers();
      }
      else
      {
         HeapTask* task = make_detached_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
         pending_tasks_.fetch_add(1, std::memory_order_relaxed);
         enqueue_external(task);
      }
   }

   template<class F, class... Args>
   auto spawn_detached_ctx(F&& f, Args&&... args)
   {
      if (curr_worker_ && curr_worker_->pool == this)
      {
         HeapTask* task =
            make_detached_spawning_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
         pending_tasks_.fetch_add(1, std::memory_order_relaxed);
         curr_worker_->local.push_bottom(task);
         wake_all_workers();
      }
      else
      {
         HeapTask* task = make_detached_spawning_task(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
         pending_tasks_.fetch_add(1, std::memory_order_relaxed);
         enqueue_external(task);
      }
   }

   template<class F, class... Args>
   auto spawn_returning(F&& f, Args&&... args)
   {
      if (curr_worker_ && curr_worker_->pool == this)
      {
         auto [task, future] =
            make_returning_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
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

   template<class F, class... Args>
   auto spawn_returning_ctx(F&& f, Args&&... args)
   {
      if (curr_worker_ && curr_worker_->pool == this)
      {
         auto [task, future] =
            make_returning_spawning_task(&curr_worker_->free_list, std::forward<F>(f), std::forward<Args>(args)...);
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

   void wait_for_tasks() noexcept
   {
      while (pending_tasks_.load(std::memory_order_acquire) != 0)
      {
         std::this_thread::yield();
      }
   }


   void worker_loop(Worker& worker)
   {
      curr_worker_ = &worker;

      constexpr unsigned spinLimit = 64;

      for (;;)
      {
         HeapTask* task = nullptr;

         if (try_get_task(worker, task))
         {
            run_heap_task(task);
            continue;
         }

         bool foundTask = false;

         for (unsigned spin = 0; spin < spinLimit; ++spin)
         {
            _mm_pause();

            if (try_get_task(worker, task))
            {
               foundTask = true;
               break;
            }
         }

         if (foundTask)
         {
            run_heap_task(task);
            continue;
         }

         if (stopping_.load(std::memory_order_acquire) && pending_tasks_.load(std::memory_order_acquire) == 0)
         {
            break;
         }

         const auto wakeSequence = worker.wakeSequence.load(std::memory_order_acquire);

         // Recheck after reading wakeSequence. This prevents a lost wake
         // if work arrived just before we attempted to sleep.
         if (try_get_task(worker, task))
         {
            run_heap_task(task);
            continue;
         }

         worker.wakeSequence.wait(wakeSequence, std::memory_order_acquire);
      }

      curr_worker_ = nullptr;
   }

   bool try_run_once(Worker& worker)
   {
      HeapTask* task = nullptr;

      if (worker.local.try_pop_bottom(task))
      {
         run_heap_task(task);
         return true;
      }

      if (try_pop_inbox(worker, worker, task))
      {
         run_heap_task(task);
         return true;
      }

      if (try_steal_work(worker, task))
      {
         run_heap_task(task);
         return true;
      }

      return false;
   }

   [[nodiscard]]
   bool try_get_task(Worker& worker, HeapTask*& task) noexcept
   {
      if (worker.local.try_pop_bottom(task))
      {
         return true;
      }

      if (try_pop_inbox(worker, worker, task))
      {
         return true;
      }

      return try_steal_work(worker, task);
   }

   [[nodiscard]]
   bool try_steal_work(Worker& self, HeapTask*& task) noexcept
   {
      if (thread_count_ <= 1)
      {
         return false;
      }

      constexpr std::size_t maxAttempts = 4;

      const std::size_t attempts = std::min<std::size_t>(maxAttempts, thread_count_ - 1);

      for (std::size_t attempt = 0; attempt < attempts; ++attempt)
      {
         std::size_t victimIndex = self.next_victim++;

         if (self.next_victim == thread_count_)
         {
            self.next_victim = 0;
         }

         if (victimIndex == self.id)
         {
            continue;
         }

         Worker& victim = *workers_[victimIndex];

         // Worker-local spawned tasks.
         if (victim.local.try_steal_top(task))
         {
            return true;
         }

         // Externally submitted work. The source queue has one producer,
         // while the owner and thieves are consumers.
         if (try_pop_inbox(self, victim, task))
         {
            return true;
         }
      }

      return false;
   }

   void stop()
   {
      stopping_.store(true, std::memory_order_release);
      wake_all_workers();
   }

   void join()
   {
      for (auto& worker : workers_)
      {
         if (worker->thread.joinable())
         {
            worker->thread.join();
         }
      }
   }

   class TaskContext
   {
      ThreadPool* pool_;

   public:
      explicit TaskContext(ThreadPool* pool)
         : pool_{pool}
      {}

      template<class F, class... Args>
      void spawn_detached(F&& f, Args&&... args)
      {
         pool_->spawn_detached(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class F, class... Args>
      void spawn_detached_ctx(F&& f, Args&&... args)
      {
         pool_->spawn_detached_ctx(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class F, class... Args>
      auto spawn_returning(F&& f, Args&&... args)
      {
         return pool_->spawn_returning(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class F, class... Args>
      auto spawn_returning_ctx(F&& f, Args&&... args)
      {
         return pool_->spawn_returning_ctx(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class R>
      R get(std::future<R>& fut)
      {
         while (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
         {
            if (!pool_->try_run_once(*curr_worker_))
            {
               std::this_thread::yield();
            }
         }

         if constexpr (std::is_void_v<R>)
         {
            fut.get();
         }
         else
         {
            return fut.get();
         }
      }
   };
};
} //namespace volt
