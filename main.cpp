#include <iostream>
#include <atomic>
#include "include/thread_pool.h"

void recursive_task(ThreadPool::TaskContext& ctx,
                    std::atomic<int>* count,
                    int depth) {
    count->fetch_add(1, std::memory_order_relaxed);

    if (depth == 0) {
        return;
    }

    ctx.spawn_detached_ctx(recursive_task, count, depth - 1);
    ctx.spawn_detached_ctx(recursive_task, count, depth - 1);
}

int returning_context_task(ThreadPool::TaskContext& ctx, int value) {
    (void)ctx;
    return value + 1;
}

int main() {
    std::cout << std::unitbuf;
    std::atomic<int> recursive_count{0};

    {
        ThreadPool pool{4};
        pool.start();

        auto f = pool.submit([](int a, int b) {
            return a + b;
        }, 4, 5);

        std::cout << "sum: " << f.get() << '\n';

        pool.submit_detached_ctx(recursive_task, &recursive_count, 3);

        auto spawned = pool.spawn_returning_ctx(returning_context_task, 41);
        std::cout << "spawned: " << spawned.get() << '\n';
    }

    std::cout << "recursive task count: " << recursive_count.load() << '\n';
}
