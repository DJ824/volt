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

   // One external producer; owner and thieves consume.
   Inbox inbox;

   // Owner pushes/pops at the bottom; other workers steal from the top.
   Local local;

   std::thread  thread;
   TaskFreeList free_list;

   std::size_t next_victim{0};

   Worker() { free_list.reserve(8192); }
};

} // namespace details
} // namespace volt
