#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "spmc.h"
#include "task_free_list.h"
#include "work_steal_deque.h"

namespace volt
{
class ThreadPool;

namespace details
{
inline constexpr std::size_t kLocalQueueSize = 8192;
inline constexpr std::size_t kInboxQueueSize = 8192;

struct HeapTask
{
   void (*run)(HeapTask*) noexcept;
   void (*destroy)(HeapTask*) noexcept;
};

struct Worker
{
   using Inbox = LockFreeQueueSpmcSeq<HeapTask*, kInboxQueueSize>;
   using Local = WorkStealDeque<HeapTask*, kLocalQueueSize>;

   alignas(64) std::atomic<std::uint32_t> wakeSequence{0};

   ThreadPool*   pool{nullptr};
   std::uint64_t id{0};

   Inbox inbox;
   Local local;

   std::thread  thread;
   TaskFreeList freeList;

   std::size_t nextVictim{0};

   Worker() { freeList.reserve(8192); }
};

} // namespace details
} // namespace volt