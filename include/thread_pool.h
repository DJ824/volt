#pragma once
#define VOLT_THREAD_POOL_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <immintrin.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
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

   int ct = 0;

   template<class F, class... Args>
   struct DetachedTask : HeapTask
   {
      std::decay_t<F>                   mFunc;
      std::tuple<std::decay_t<Args>...> mArgs;

      DetachedTask(void (*destroyFn)(HeapTask*) noexcept, F&& f, Args&&... args)
         : HeapTask{&runImpl, destroyFn}
         , mFunc{std::forward<F>(f)}
         , mArgs{std::forward<Args>(args)...}
      {}


      static void runImpl(HeapTask* base) noexcept
      {
         auto self = static_cast<DetachedTask*>(base);
         std::apply(self->mFunc, self->mArgs);
      }
   };

   template<class F, class... Args>
   struct DetachedContextTask : HeapTask
   {
      ThreadPool*                       mPool;
      std::decay_t<F>                   mFunc;
      std::tuple<std::decay_t<Args>...> mArgs;

      DetachedContextTask(void (*destroyFn)(HeapTask*) noexcept, ThreadPool* p, F&& f, Args&&... args)
         : HeapTask{&runImpl, destroyFn}
         , mPool(p)
         , mFunc(std::forward<F>(f))
         , mArgs(std::forward<Args>(args)...)
      {}

      static void runImpl(HeapTask* base) noexcept
      {
         auto self = static_cast<DetachedContextTask*>(base);

         TaskContext ctx{self->mPool};
         std::apply([&](auto&&... unpacked) { self->mFunc(ctx, std::forward<decltype(unpacked)>(unpacked)...); },
                    self->mArgs);
      }
   };

   template<class F, class... Args>
   struct ReturningTask : HeapTask
   {
      using R = std::invoke_result_t<F, Args...>;

      std::decay_t<F>                   mFunc;
      std::tuple<std::decay_t<Args>...> mArgs;
      std::promise<R>                   mPromise;

      ReturningTask(void (*destroyFn)(HeapTask*) noexcept, F&& f, Args&&... args)
         : HeapTask{&runImpl, destroyFn}
         , mFunc{std::forward<F>(f)}
         , mArgs{std::forward<Args>(args)...}
      {}

      static void runImpl(HeapTask* base) noexcept
      {
         auto* self = static_cast<ReturningTask*>(base);
         try
         {
            if constexpr (std::is_void_v<R>)
            {
               std::apply(self->mFunc, self->mArgs);
               self->mPromise.set_value();
            }
            else
            {
               self->mPromise.set_value(std::apply(self->mFunc, self->mArgs));
            }
         }
         catch (...)
         {
            self->mPromise.set_exception(std::current_exception());
         }
      }
   };

   template<class F, class... Args>
   struct ReturningContextTask : HeapTask
   {
      using R = std::invoke_result_t<F, TaskContext&, Args...>;
      ThreadPool*                       mPool;
      std::decay_t<F>                   mFunc;
      std::tuple<std::decay_t<Args>...> mArgs;
      std::promise<R>                   mPromise;

      ReturningContextTask(void (*destroyFn)(HeapTask*) noexcept, ThreadPool* p, F&& f, Args&&... args)
         : HeapTask{&runImpl, destroyFn}
         , mPool(p)
         , mFunc(std::forward<F>(f))
         , mArgs(std::forward<Args>(args)...)
      {}

      static void runImpl(HeapTask* base) noexcept
      {
         auto self = static_cast<ReturningContextTask*>(base);

         try
         {
            TaskContext ctx{self->mPool};

            if constexpr (std::is_void_v<R>)
            {
               std::apply([&](auto&&... unpacked) { self->mFunc(ctx, std::forward<decltype(unpacked)>(unpacked)...); },
                          self->mArgs);
               self->mPromise.set_value();
            }
            else
               self->mPromise.set_value(
                  std::apply([&](auto&&... unpacked)
                             { return self->mFunc(ctx, std::forward<decltype(unpacked)>(unpacked)...); },
                             self->mArgs));
         }
         catch (...)
         {
            self->mPromise.set_exception(std::current_exception());
         }
      }
   };

   template<class T>
   static void heapDestroy(HeapTask* base) noexcept
   {
      delete static_cast<T*>(base);
   }

   template<class T>
   static void pooledDestroy(HeapTask* base) noexcept
   {
      auto* self    = static_cast<T*>(base);
      void* storage = self;
      self->~T();
      mCurrentWorker->freeList.release(storage);
   }

   template<class T, class... Args>
   T* makeTask(TaskFreeList* freeList, Args&&... args)
   {
      if constexpr (sizeof(T) <= TaskFreeList::block_size() && alignof(T) <= TaskFreeList::block_align())
      {
         if (freeList != nullptr)
         {
            void* storage = freeList->acquire();
            return new (storage) T(&pooledDestroy<T>, std::forward<Args>(args)...);
         }
      }

      return new T(&heapDestroy<T>, std::forward<Args>(args)...);
   }

   template<class F, class... Args>
   auto makeReturningTask(TaskFreeList* freeList, F&& f, Args&&... args)
   {
      using T = ReturningTask<F, Args...>;
      using R = typename T::R;

      auto* task = makeTask<T>(freeList, std::forward<F>(f), std::forward<Args>(args)...);

      std::future<R> future = task->mPromise.get_future();
      return std::pair<HeapTask*, std::future<R>>{task, std::move(future)};
   }

   template<class F, class... Args>
   auto makeReturningSpawningTask(TaskFreeList* freeList, F&& f, Args&&... args)
   {
      using T = ReturningContextTask<F, Args...>;
      using R = typename T::R;

      auto*          task   = makeTask<T>(freeList, this, std::forward<F>(f), std::forward<Args>(args)...);
      std::future<R> future = task->mPromise.get_future();
      return std::pair<HeapTask*, std::future<R>>{task, std::move(future)};
   }

   template<class F, class... Args>
   auto makeDetachedTask(TaskFreeList* freeList, F&& f, Args&&... args)
   {
      using T = DetachedTask<F, Args...>;

      return makeTask<T>(freeList, std::forward<F>(f), std::forward<Args>(args)...);
   }

   template<class F, class... Args>
   auto makeDetachedSpawningTask(TaskFreeList* freeList, F&& f, Args&&... args)
   {
      using T = DetachedContextTask<F, Args...>;

      return makeTask<T>(freeList, this, std::forward<F>(f), std::forward<Args>(args)...);
   }


   inline static thread_local Worker* mCurrentWorker = nullptr;

   bool                                 mStarted{false};
   std::size_t                          mThreadCount{0};
   std::vector<std::unique_ptr<Worker>> mWorkers;

   std::vector<std::vector<InboxToken>> mInboxTokens;

   std::atomic<bool>        mStopping{false};
   std::size_t              mSubmitCursor{0};
   std::atomic<std::size_t> mPendingTasks{0};

   std::vector<std::uint32_t> mWorkerCpuSetIds;
   TaskFreeList               mGlobalFreeList;

   void runHeapTask(HeapTask* task) noexcept
   {
      task->run(task);
      task->destroy(task);

      const auto previous = mPendingTasks.fetch_sub(1, std::memory_order_release);

      if (previous == 1 && mStopping.load(std::memory_order_acquire))
      {
         wakeAllWorkers();
      }
   }

   void wakeAllWorkers() noexcept
   {
      for (auto& worker : mWorkers)
      {
         worker->wakeSequence.fetch_add(1, std::memory_order_release);

         worker->wakeSequence.notify_one();
      }
   }

   [[nodiscard]]
   static std::size_t defaultThreadCount() noexcept
   {
      const auto count = std::thread::hardware_concurrency();
      return count == 0 ? 1 : static_cast<std::size_t>(count);
   }

#ifdef _WIN32
   struct CpuSetDescriptor
   {
      std::uint32_t mId{0};
      std::uint16_t mGroup{0};
      std::uint8_t  mLogicalProcessor{0};
      std::uint8_t  mCore{0};
      std::uint8_t  mEfficiencyClass{0};
   };

   struct CpuCoreDescriptor
   {
      std::uint16_t                 mGroup{0};
      std::uint8_t                  mCore{0};
      std::uint8_t                  mEfficiencyClass{0};
      std::vector<CpuSetDescriptor> mLogicalProcessors;
   };

   [[nodiscard]]
   static std::vector<CpuSetDescriptor> queryCpuSets()
   {
      ULONG requiredSize = 0;

      static_cast<void>(GetSystemCpuSetInformation(nullptr, 0, &requiredSize, GetCurrentProcess(), 0));

      if (requiredSize == 0)
      {
         return {};
      }

      std::vector<std::byte> buffer(requiredSize);
      ULONG                  returnedSize = requiredSize;

      if (GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
                                     requiredSize,
                                     &returnedSize,
                                     GetCurrentProcess(),
                                     0) == FALSE)
      {
         return {};
      }

      std::vector<CpuSetDescriptor> result;
      std::size_t                   offset = 0;

      while (offset < returnedSize)
      {
         const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);

         if (info->Size == 0 || offset + info->Size > returnedSize)
         {
            break;
         }

         if (info->Type == CpuSetInformation)
         {
            const auto& cpu = info->CpuSet;

            if (!(cpu.Allocated && !cpu.AllocatedToTargetProcess))
            {
               result.push_back(CpuSetDescriptor{static_cast<std::uint32_t>(cpu.Id),
                                                 static_cast<std::uint16_t>(cpu.Group),
                                                 static_cast<std::uint8_t>(cpu.LogicalProcessorIndex),
                                                 static_cast<std::uint8_t>(cpu.CoreIndex),
                                                 static_cast<std::uint8_t>(cpu.EfficiencyClass)});
            }
         }

         offset += info->Size;
      }

      return result;
   }

   [[nodiscard]]
   static std::vector<CpuCoreDescriptor> groupCpuSetsByCore(std::vector<CpuSetDescriptor> cpuSets)
   {
      std::sort(cpuSets.begin(),
                cpuSets.end(),
                [](const CpuSetDescriptor& lhs, const CpuSetDescriptor& rhs)
                {
                   if (lhs.mEfficiencyClass != rhs.mEfficiencyClass)
                   {
                      return lhs.mEfficiencyClass > rhs.mEfficiencyClass;
                   }

                   if (lhs.mGroup != rhs.mGroup)
                   {
                      return lhs.mGroup < rhs.mGroup;
                   }

                   if (lhs.mCore != rhs.mCore)
                   {
                      return lhs.mCore < rhs.mCore;
                   }

                   return lhs.mLogicalProcessor < rhs.mLogicalProcessor;
                });

      std::vector<CpuCoreDescriptor> cores;

      for (const CpuSetDescriptor& cpu : cpuSets)
      {
         auto coreIt = std::find_if(cores.begin(),
                                    cores.end(),
                                    [&](const CpuCoreDescriptor& core)
                                    { return core.mGroup == cpu.mGroup && core.mCore == cpu.mCore; });

         if (coreIt == cores.end())
         {
            cores.push_back(CpuCoreDescriptor{cpu.mGroup, cpu.mCore, cpu.mEfficiencyClass, {}});

            coreIt = std::prev(cores.end());
         }

         coreIt->mLogicalProcessors.push_back(cpu);
      }

      return cores;
   }

   [[nodiscard]]
   static std::vector<std::uint8_t> efficiencyClassesDescending(const std::vector<CpuCoreDescriptor>& cores)
   {
      std::vector<std::uint8_t> classes;
      classes.reserve(cores.size());

      for (const CpuCoreDescriptor& core : cores)
      {
         if (std::find(classes.begin(), classes.end(), core.mEfficiencyClass) == classes.end())
         {
            classes.push_back(core.mEfficiencyClass);
         }
      }

      std::sort(classes.begin(), classes.end(), std::greater<>{});
      return classes;
   }

   static void appendPrimaryLogicalProcessors(const std::vector<CpuCoreDescriptor>& cores,
                                              std::uint8_t                          efficiencyClass,
                                              std::vector<std::uint32_t>&           output)
   {
      for (const CpuCoreDescriptor& core : cores)
      {
         if (core.mEfficiencyClass == efficiencyClass && !core.mLogicalProcessors.empty())
         {
            output.push_back(core.mLogicalProcessors.front().mId);
         }
      }
   }

   static void appendSmtSiblings(const std::vector<CpuCoreDescriptor>& cores,
                                 std::uint8_t                          efficiencyClass,
                                 std::vector<std::uint32_t>&           output)
   {
      for (const CpuCoreDescriptor& core : cores)
      {
         if (core.mEfficiencyClass != efficiencyClass)
         {
            continue;
         }

         for (std::size_t logical = 1; logical < core.mLogicalProcessors.size(); ++logical)
         {
            output.push_back(core.mLogicalProcessors[logical].mId);
         }
      }
   }

   [[nodiscard]]
   static std::vector<std::uint32_t> buildWorkerCpuSetOrder() noexcept
   {
      try
      {
         auto cores = groupCpuSetsByCore(queryCpuSets());

         if (cores.empty())
         {
            return {};
         }

         const auto classes = efficiencyClassesDescending(cores);

         if (classes.empty())
         {
            return {};
         }

         std::vector<std::uint32_t> order;

         std::size_t logicalProcessorCount = 0;
         for (const CpuCoreDescriptor& core : cores)
         {
            logicalProcessorCount += core.mLogicalProcessors.size();
         }
         order.reserve(logicalProcessorCount);

         appendPrimaryLogicalProcessors(cores, classes.front(), order);

         if (classes.size() >= 2)
         {
            appendPrimaryLogicalProcessors(cores, classes[1], order);
         }

         appendSmtSiblings(cores, classes.front(), order);

         for (std::size_t classIndex = 2; classIndex < classes.size(); ++classIndex)
         {
            appendPrimaryLogicalProcessors(cores, classes[classIndex], order);
         }

         for (std::size_t classIndex = 1; classIndex < classes.size(); ++classIndex)
         {
            appendSmtSiblings(cores, classes[classIndex], order);
         }

         return order;
      }
      catch (...)
      {
         return {};
      }
   }
#endif

   void pinCurrentThread(std::size_t workerId) const noexcept
   {
#ifdef _WIN32
      if (workerId >= mWorkerCpuSetIds.size())
      {
         return;
      }

      const ULONG cpuSetId = static_cast<ULONG>(mWorkerCpuSetIds[workerId]);

      static_cast<void>(SetThreadSelectedCpuSets(GetCurrentThread(), &cpuSetId, 1));
#else
      static_cast<void>(workerId);
#endif
   }

   [[nodiscard]]
   bool tryPopInbox(Worker& consumer, Worker& source, HeapTask*& task) noexcept
   {
      const auto consumerId = static_cast<std::size_t>(consumer.id);
      const auto sourceId   = static_cast<std::size_t>(source.id);

      return source.inbox.tryPop(mInboxTokens[consumerId][sourceId], task);
   }

   void enqueueExternal(std::size_t workerId, HeapTask* task)
   {
      Worker& worker = *mWorkers[workerId];

      worker.inbox.push(task);

      worker.wakeSequence.fetch_add(1, std::memory_order_release);

      worker.wakeSequence.notify_one();
   }

   void enqueueExternal(HeapTask* task)
   {
      auto threadId = mSubmitCursor++ % mThreadCount;
      enqueueExternal(threadId, task);
   }

public:
   ThreadPool()
      : ThreadPool(2)
   {}

   explicit ThreadPool(std::size_t threadCount)
      : mThreadCount{std::max<std::size_t>(1, threadCount)}
   {
#ifdef _WIN32
      mWorkerCpuSetIds = buildWorkerCpuSetOrder();
#endif

      mWorkers.reserve(mThreadCount);

      for (std::size_t i = 0; i < mThreadCount; ++i)
      {
         auto worker        = std::make_unique<Worker>();
         worker->pool       = this;
         worker->id         = i;
         worker->nextVictim = (i + 1) % mThreadCount;
         mWorkers.push_back(std::move(worker));
      }

      mInboxTokens.reserve(mThreadCount);

      for (std::size_t consumer = 0; consumer < mThreadCount; ++consumer)
      {
         mInboxTokens.emplace_back();
         auto& tokens = mInboxTokens.back();
         tokens.reserve(mThreadCount);

         for (std::size_t source = 0; source < mThreadCount; ++source)
         {
            tokens.emplace_back(mWorkers[source]->inbox.makeConsumerToken());
         }
      }

      mGlobalFreeList.allocate_chunk();
   }

   ~ThreadPool()
   {
      stop();
      join();
   }

   void start()
   {
      if (mStarted)
      {
         return;
      }

      mStarted = true;

      for (auto& worker : mWorkers)
      {
         auto workerPtr = worker.get();
         worker->thread = std::thread(
            [this, workerPtr]
            {
               pinCurrentThread(static_cast<std::size_t>(workerPtr->id));
               workerLoop(*workerPtr);
            });
      }
   }

   template<class F, class... Args>
   auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
   {
      auto  workerId      = mSubmitCursor++ % mThreadCount;
      auto* worker        = mWorkers[workerId].get();
      auto [task, future] = makeReturningTask(&mGlobalFreeList, std::forward<F>(f), std::forward<Args>(args)...);
      mPendingTasks.fetch_add(1, std::memory_order_relaxed);
      enqueueExternal(workerId, task);
      return std::move(future);
   }

   template<class F, class... Args>
   auto submitDetached(F&& f, Args&&... args)
   {
      const auto workerId = mSubmitCursor++ % mThreadCount;
      HeapTask*  task     = makeDetachedTask(&mGlobalFreeList, std::forward<F>(f), std::forward<Args>(args)...);
      mPendingTasks.fetch_add(1, std::memory_order_relaxed);
      enqueueExternal(workerId, task);
   }

   template<class F, class... Args>
   auto submitDetachedCtx(F&& f, Args&&... args)
   {
      auto      workerId = mSubmitCursor++ % mThreadCount;
      auto*     worker   = mWorkers[workerId].get();
      HeapTask* task     = makeDetachedSpawningTask(&mGlobalFreeList, std::forward<F>(f), std::forward<Args>(args)...);

      mPendingTasks.fetch_add(1, std::memory_order_seq_cst);
      enqueueExternal(workerId, task);
   }

   template<class F, class... Args>
   auto spawnDetached(F&& f, Args&&... args)
   {
      if (mCurrentWorker && mCurrentWorker->pool == this)
      {
         HeapTask* task = makeDetachedTask(&mCurrentWorker->freeList, std::forward<F>(f), std::forward<Args>(args)...);
         mPendingTasks.fetch_add(1, std::memory_order_relaxed);
         mCurrentWorker->local.pushBottom(task);
         wakeAllWorkers();
      }
      else
      {
         HeapTask* task = makeDetachedTask(&mGlobalFreeList, std::forward<F>(f), std::forward<Args>(args)...);
         mPendingTasks.fetch_add(1, std::memory_order_relaxed);
         enqueueExternal(task);
      }
   }

   template<class F, class... Args>
   auto spawnDetachedCtx(F&& f, Args&&... args)
   {
      if (mCurrentWorker && mCurrentWorker->pool == this)
      {
         HeapTask* task =
            makeDetachedSpawningTask(&mCurrentWorker->freeList, std::forward<F>(f), std::forward<Args>(args)...);
         mPendingTasks.fetch_add(1, std::memory_order_relaxed);
         mCurrentWorker->local.pushBottom(task);
         wakeAllWorkers();
      }
      else
      {
         HeapTask* task = makeDetachedSpawningTask(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
         mPendingTasks.fetch_add(1, std::memory_order_relaxed);
         enqueueExternal(task);
      }
   }

   template<class F, class... Args>
   auto spawnReturning(F&& f, Args&&... args)
   {
      if (mCurrentWorker && mCurrentWorker->pool == this)
      {
         auto [task, future] =
            makeReturningTask(&mCurrentWorker->freeList, std::forward<F>(f), std::forward<Args>(args)...);
         mPendingTasks.fetch_add(1, std::memory_order_seq_cst);
         mCurrentWorker->local.pushBottom(task);
         wakeAllWorkers();
         return std::move(future);
      }

      auto [task, future] = makeReturningTask(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
      mPendingTasks.fetch_add(1, std::memory_order_seq_cst);
      enqueueExternal(task);
      return std::move(future);
   }

   template<class F, class... Args>
   auto spawnReturningCtx(F&& f, Args&&... args)
   {
      if (mCurrentWorker && mCurrentWorker->pool == this)
      {
         auto [task, future] =
            makeReturningSpawningTask(&mCurrentWorker->freeList, std::forward<F>(f), std::forward<Args>(args)...);
         mPendingTasks.fetch_add(1, std::memory_order_seq_cst);
         mCurrentWorker->local.pushBottom(task);
         wakeAllWorkers();
         return std::move(future);
      }

      auto [task, future] = makeReturningSpawningTask(nullptr, std::forward<F>(f), std::forward<Args>(args)...);
      mPendingTasks.fetch_add(1, std::memory_order_seq_cst);
      enqueueExternal(task);
      return std::move(future);
   }

   void waitForTasks() noexcept
   {
      while (mPendingTasks.load(std::memory_order_acquire) != 0)
      {
         std::this_thread::yield();
      }
   }


   void workerLoop(Worker& worker)
   {
      mCurrentWorker = &worker;

      constexpr unsigned spinLimit = 64;

      for (;;)
      {
         HeapTask* task = nullptr;

         if (tryGetTask(worker, task))
         {
            runHeapTask(task);
            continue;
         }

         bool foundTask = false;

         for (unsigned spin = 0; spin < spinLimit; ++spin)
         {
            _mm_pause();

            if (tryGetTask(worker, task))
            {
               foundTask = true;
               break;
            }
         }

         if (foundTask)
         {
            runHeapTask(task);
            continue;
         }

         if (mStopping.load(std::memory_order_acquire) && mPendingTasks.load(std::memory_order_acquire) == 0)
         {
            break;
         }

         const auto wakeSequence = worker.wakeSequence.load(std::memory_order_acquire);

         if (tryGetTask(worker, task))
         {
            runHeapTask(task);
            continue;
         }

         worker.wakeSequence.wait(wakeSequence, std::memory_order_acquire);
      }

      mCurrentWorker = nullptr;
   }

   bool tryRunOnce(Worker& worker)
   {
      HeapTask* task = nullptr;

      if (worker.local.tryPopBottom(task))
      {
         runHeapTask(task);
         return true;
      }

      if (tryPopInbox(worker, worker, task))
      {
         runHeapTask(task);
         return true;
      }

      if (tryStealWork(worker, task))
      {
         runHeapTask(task);
         return true;
      }

      return false;
   }

   [[nodiscard]]
   bool tryGetTask(Worker& worker, HeapTask*& task) noexcept
   {
      if (worker.local.tryPopBottom(task))
      {
         return true;
      }

      if (tryPopInbox(worker, worker, task))
      {
         return true;
      }

      return tryStealWork(worker, task);
   }

   [[nodiscard]]
   bool tryStealWork(Worker& self, HeapTask*& task) noexcept
   {
      if (mThreadCount <= 1)
      {
         return false;
      }

      constexpr std::size_t maxAttempts = 4;

      const std::size_t attempts = std::min<std::size_t>(maxAttempts, mThreadCount - 1);

      for (std::size_t attempt = 0; attempt < attempts; ++attempt)
      {
         std::size_t victimIndex = self.nextVictim++;

         if (self.nextVictim == mThreadCount)
         {
            self.nextVictim = 0;
         }

         if (victimIndex == self.id)
         {
            continue;
         }

         Worker& victim = *mWorkers[victimIndex];

         if (victim.local.tryStealTop(task))
         {
            return true;
         }

         if (tryPopInbox(self, victim, task))
         {
            return true;
         }
      }

      return false;
   }

   void stop()
   {
      mStopping.store(true, std::memory_order_release);
      wakeAllWorkers();
   }

   void join()
   {
      for (auto& worker : mWorkers)
      {
         if (worker->thread.joinable())
         {
            worker->thread.join();
         }
      }
   }

   class TaskContext
   {
      ThreadPool* mPool;

   public:
      explicit TaskContext(ThreadPool* pool)
         : mPool{pool}
      {}

      template<class F, class... Args>
      void spawnDetached(F&& f, Args&&... args)
      {
         mPool->spawnDetached(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class F, class... Args>
      void spawnDetachedCtx(F&& f, Args&&... args)
      {
         mPool->spawnDetachedCtx(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class F, class... Args>
      auto spawnReturning(F&& f, Args&&... args)
      {
         return mPool->spawnReturning(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class F, class... Args>
      auto spawnReturningCtx(F&& f, Args&&... args)
      {
         return mPool->spawnReturningCtx(std::forward<F>(f), std::forward<Args>(args)...);
      }

      template<class R>
      R get(std::future<R>& fut)
      {
         while (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
         {
            if (!mPool->tryRunOnce(*mCurrentWorker))
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