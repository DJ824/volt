#include <nanobench.h>
#include <thread_pool/thread_pool.h>

#include <BS_thread_pool_light.hpp>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <function2/function2.hpp>
#include <functional>
#include <future>
#include <iostream>
#include <string>
#include <task_thread_pool.hpp>
#include <thread>
#include <vector>

#include "thread_pool.h"

#ifndef RECURSIVE_RESULTS_MARKDOWN_FILE
#define RECURSIVE_RESULTS_MARKDOWN_FILE "recursive_benchmark_results.md"
#endif

std::size_t benchmark_thread_count() {
    const auto count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : count;
}

std::size_t bounded_spawn_depth(std::size_t thread_count) {
    std::size_t depth = 0;
    while (thread_count > 1) {
        thread_count >>= 1;
        ++depth;
    }
    return std::min<std::size_t>(depth, 3);
}

std::vector<int> make_input(std::size_t size) {
    std::vector<int> values(size);
    std::uint32_t state = 0x12345678u;

    for (auto& value : values) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = static_cast<int>(state);
    }

    return values;
}

void merge_range(std::vector<int>& values, std::vector<int>& scratch,
                 std::size_t lo, std::size_t mid, std::size_t hi) {
    std::merge(values.begin() + static_cast<std::ptrdiff_t>(lo),
               values.begin() + static_cast<std::ptrdiff_t>(mid),
               values.begin() + static_cast<std::ptrdiff_t>(mid),
               values.begin() + static_cast<std::ptrdiff_t>(hi),
               scratch.begin() + static_cast<std::ptrdiff_t>(lo));

    std::copy(scratch.begin() + static_cast<std::ptrdiff_t>(lo),
              scratch.begin() + static_cast<std::ptrdiff_t>(hi),
              values.begin() + static_cast<std::ptrdiff_t>(lo));
}

void serial_sort(std::vector<int>& values, std::size_t lo, std::size_t hi) {
    std::sort(values.begin() + static_cast<std::ptrdiff_t>(lo),
              values.begin() + static_cast<std::ptrdiff_t>(hi));
}

void volt_merge_sort(ThreadPool::TaskContext& ctx, std::vector<int>& values,
                     std::vector<int>& scratch, std::size_t lo, std::size_t hi,
                     std::size_t cutoff, std::size_t depth) {
    if (hi - lo <= cutoff || depth == 0) {
        serial_sort(values, lo, hi);
        return;
    }

    const std::size_t mid = lo + (hi - lo) / 2;
    auto left = ctx.spawn_returning_ctx(volt_merge_sort, std::ref(values), std::ref(scratch),
                                        lo, mid, cutoff, depth - 1);

    volt_merge_sort(ctx, values, scratch, mid, hi, cutoff, depth - 1);
    ctx.get(left);
    merge_range(values, scratch, lo, mid, hi);
}

template <typename Pool, typename Submit>
void future_merge_sort(Pool& pool, Submit& submit, std::vector<int>& values,
                       std::vector<int>& scratch, std::size_t lo, std::size_t hi,
                       std::size_t cutoff, std::size_t depth) {
    if (hi - lo <= cutoff || depth == 0) {
        serial_sort(values, lo, hi);
        return;
    }

    const std::size_t mid = lo + (hi - lo) / 2;
    auto left = submit([&pool, &submit, &values, &scratch, lo, mid, cutoff, depth] {
        future_merge_sort(pool, submit, values, scratch, lo, mid, cutoff, depth - 1);
    });

    future_merge_sort(pool, submit, values, scratch, mid, hi, cutoff, depth - 1);
    left.get();
    merge_range(values, scratch, lo, mid, hi);
}

template <typename Runner>
void run_recursive_benchmark(ankerl::nanobench::Bench* bench, const std::vector<int>& input,
                             std::size_t cutoff, std::size_t depth, const char* name,
                             Runner&& runner) {
    bench->run(name, [&] {
        std::vector<int> values = input;
        std::vector<int> scratch(values.size());

        runner(values, scratch, cutoff, depth);
        ankerl::nanobench::doNotOptimizeAway(values.data());
    });
}

int main() {
    using namespace std::chrono_literals;

    const auto thread_count = benchmark_thread_count();
    const auto depth = bounded_spawn_depth(thread_count);

    const std::vector<std::pair<std::size_t, std::size_t>> args = {
        {1u << 18, 4096},
        {1u << 20, 8192},
        {1u << 22, 16384},
        {10'000'000, 32768},
    };

    std::ofstream output_file(RECURSIVE_RESULTS_MARKDOWN_FILE, std::ios::out);

    for (const auto& [size, cutoff] : args) {
        std::cerr << "starting recursive merge sort with " << size
                  << " values, cutoff " << cutoff << ", depth " << depth << '\n';

        const auto input = make_input(size);
        ankerl::nanobench::Bench bench;
        auto bench_title = std::string("recursive merge sort ") + std::to_string(size) +
                           " values";

        bench.title(bench_title)
            .warmup(5)
            .relative(true)
            .minEpochIterations(10)
            .output(&output_file);
        bench.timeUnit(1ms, "ms");

        {
            std::cerr << "  volt::ThreadPool\n";
            ThreadPool pool{thread_count};
            pool.start();
            auto submit = [&](auto&& task) {
                return pool.submit(std::forward<decltype(task)>(task));
            };

            run_recursive_benchmark(&bench, input, cutoff, depth, "volt::ThreadPool",
                                    [&](std::vector<int>& values, std::vector<int>& scratch,
                                        std::size_t task_cutoff, std::size_t task_depth) {
                                        auto done = submit([&] {
                                            future_merge_sort(pool, submit, values, scratch, 0,
                                                              values.size(), task_cutoff, task_depth);
                                        });
                                        done.get();
                                    });
        }

        {
            std::cerr << "  dp::thread_pool - std::function\n";
            dp::thread_pool<std::function<void()>> pool{static_cast<unsigned int>(thread_count)};
            auto submit = [&](auto&& task) {
                return pool.enqueue(std::forward<decltype(task)>(task));
            };

            run_recursive_benchmark(&bench, input, cutoff, depth,
                                    "dp::thread_pool - std::function",
                                    [&](std::vector<int>& values, std::vector<int>& scratch,
                                        std::size_t task_cutoff, std::size_t task_depth) {
                                        auto done = submit([&] {
                                            future_merge_sort(pool, submit, values, scratch, 0,
                                                              values.size(), task_cutoff, task_depth);
                                        });
                                        done.get();
                                    });
        }

#ifdef __cpp_lib_move_only_function
        {
            std::cerr << "  dp::thread_pool - std::move_only_function\n";
            dp::thread_pool pool{static_cast<unsigned int>(thread_count)};
            auto submit = [&](auto&& task) {
                return pool.enqueue(std::forward<decltype(task)>(task));
            };

            run_recursive_benchmark(&bench, input, cutoff, depth,
                                    "dp::thread_pool - std::move_only_function",
                                    [&](std::vector<int>& values, std::vector<int>& scratch,
                                        std::size_t task_cutoff, std::size_t task_depth) {
                                        auto done = submit([&] {
                                            future_merge_sort(pool, submit, values, scratch, 0,
                                                              values.size(), task_cutoff, task_depth);
                                        });
                                        done.get();
                                    });
        }
#endif

        {
            std::cerr << "  dp::thread_pool - fu2::unique_function\n";
            dp::thread_pool<fu2::unique_function<void()>> pool{static_cast<unsigned int>(thread_count)};
            auto submit = [&](auto&& task) {
                return pool.enqueue(std::forward<decltype(task)>(task));
            };

            run_recursive_benchmark(&bench, input, cutoff, depth,
                                    "dp::thread_pool - fu2::unique_function",
                                    [&](std::vector<int>& values, std::vector<int>& scratch,
                                        std::size_t task_cutoff, std::size_t task_depth) {
                                        auto done = submit([&] {
                                            future_merge_sort(pool, submit, values, scratch, 0,
                                                              values.size(), task_cutoff, task_depth);
                                        });
                                        done.get();
                                    });
        }

        {
            std::cerr << "  BS::thread_pool\n";
            BS::thread_pool_light pool{static_cast<unsigned int>(thread_count)};
            auto submit = [&](auto&& task) {
                return pool.submit(std::forward<decltype(task)>(task));
            };

            run_recursive_benchmark(&bench, input, cutoff, depth, "BS::thread_pool",
                                    [&](std::vector<int>& values, std::vector<int>& scratch,
                                        std::size_t task_cutoff, std::size_t task_depth) {
                                        auto done = submit([&] {
                                            future_merge_sort(pool, submit, values, scratch, 0,
                                                              values.size(), task_cutoff, task_depth);
                                        });
                                        done.get();
                                    });
        }

        {
            std::cerr << "  task_thread_pool\n";
            task_thread_pool::task_thread_pool pool{static_cast<unsigned int>(thread_count)};
            auto submit = [&](auto&& task) {
                return pool.submit(std::forward<decltype(task)>(task));
            };

            run_recursive_benchmark(&bench, input, cutoff, depth, "task_thread_pool",
                                    [&](std::vector<int>& values, std::vector<int>& scratch,
                                        std::size_t task_cutoff, std::size_t task_depth) {
                                        auto done = submit([&] {
                                            future_merge_sort(pool, submit, values, scratch, 0,
                                                              values.size(), task_cutoff, task_depth);
                                        });
                                        done.get();
                                    });
        }

        std::cerr << "finished recursive merge sort with " << size << " values\n";
    }

    std::cout << "wrote " << RECURSIVE_RESULTS_MARKDOWN_FILE << '\n';
}
