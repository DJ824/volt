#include <nanobench.h>
#include <thread_pool/thread_pool.h>

#include <BS_thread_pool_light.hpp>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_group.h>
#include <quickpool.hpp>
#include <thread_pool.hpp>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <fstream>
#include <function2/function2.hpp>
#include <functional>
#include <future>
#include <iostream>
#include <riften/thiefpool.hpp>
#include <string>
#include <task_thread_pool.hpp>
#include <thread>
#include <type_traits>
#include <vector>

#include "thread_pool.h"
#include "utilities.h"

#ifndef RESULTS_MARKDOWN_FILE
#define RESULTS_MARKDOWN_FILE "thread_pool_benchmark_results.md"
#endif

std::size_t benchmark_thread_count() {
    const auto count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : count;
}

template <std::integral DataType, typename FutureProvider>
    requires std::same_as<std::future<void>,
                          std::invoke_result_t<FutureProvider&, const std::vector<DataType>&,
                                               const std::vector<DataType>&>>
void run_benchmark(ankerl::nanobench::Bench* bench, const std::size_t array_size,
                   const std::size_t multiplications_to_perform, const char* name,
                   FutureProvider&& provider) {
    const std::vector<DataType> a(array_size, 2);
    const std::vector<DataType> b(array_size, 3);

    std::vector<std::future<void>> results;
    results.reserve(multiplications_to_perform);

    bench->run(name, [&]() {
        results.clear();

        for (std::size_t i = 0; i < multiplications_to_perform; ++i) {
            results.emplace_back(provider(a, b));
        }

        for (auto& fut : results) {
            fut.wait();
        }
    });
}


int main() {
    using namespace std::chrono_literals;
    const auto thread_count = benchmark_thread_count();

    // task that runs on a new thread (potentially)
    auto thread_task = [](const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result(a.size());
        multiply_array(a, b, result);
        ankerl::nanobench::doNotOptimizeAway(result);
    };

    std::vector<std::pair<std::size_t, std::size_t>> args = {
        {8, 100'000}, {64, 75'000}, {256, 50'000}, {512, 35'000}, {1024, 25'000}};

    std::ofstream output_file(RESULTS_MARKDOWN_FILE, std::ios::out);

    for (const auto& [array_size, iterations] : args) {
        std::cerr << "starting matrix multiplication " << array_size << "x" << array_size
                  << " with " << iterations << " submissions\n";
        ankerl::nanobench::Bench bench;
        auto bench_title = std::string("matrix multiplication ") + std::to_string(array_size) +
                           "x" + std::to_string(array_size);
        bench.title(bench_title)
            .warmup(10)
            .relative(true)
            .minEpochIterations(15)
            .output(&output_file);
        bench.timeUnit(1ms, "ms");

        {
            std::cerr << "  volt::ThreadPool\n";
            ThreadPool pool{thread_count};
            pool.start();
            run_benchmark<int>(
                &bench, array_size, iterations, "volt::ThreadPool",
                [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                    return pool.submit(thread_task, std::cref(a), std::cref(b));
                });
        }

        {
            std::cerr << "  dp::thread_pool - std::function\n";
            dp::thread_pool<std::function<void()>> pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations, "dp::thread_pool - std::function",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return pool.enqueue(thread_task, std::cref(a), std::cref(b));
                               });
        }

#ifdef __cpp_lib_move_only_function
        {
            std::cerr << "  dp::thread_pool - std::move_only_function\n";
            dp::thread_pool pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations,
                               "dp::thread_pool - std::move_only_function",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return pool.enqueue(thread_task, std::cref(a), std::cref(b));
                               });
        }
#endif
        {
            std::cerr << "  dp::thread_pool - fu2::unique_function\n";
            dp::thread_pool<fu2::unique_function<void()>> pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations,
                               "dp::thread_pool - fu2::unique_function",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return pool.enqueue(thread_task, std::cref(a), std::cref(b));
                               });
        }

        {
            std::cerr << "  BS::thread_pool\n";
            BS::thread_pool_light bs_thread_pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations, "BS::thread_pool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return bs_thread_pool.submit(thread_task, std::cref(a), std::cref(b));
                               });
        }

        {
            std::cerr << "  task_thread_pool\n";
            task_thread_pool::task_thread_pool ttp{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations, "task_thread_pool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return ttp.submit(thread_task, std::cref(a), std::cref(b));
                               });
        }

        {
            std::cerr << "  riften::Thiefpool\n";
            riften::Thiefpool riften_thiefpool{thread_count};
            run_benchmark<int>(&bench, array_size, iterations, "riften::Thiefpool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return riften_thiefpool.enqueue(thread_task, std::cref(a), std::cref(b));
                               });
        }

        {
            std::cerr << "  quickpool::ThreadPool\n";
            quickpool::ThreadPool pool{thread_count};
            run_benchmark<int>(&bench, array_size, iterations, "quickpool::ThreadPool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return pool.async(thread_task, std::cref(a), std::cref(b));
                               });
        }

        {
            std::cerr << "  HQarroum::thread_pool\n";
            thread::pool::pool_t pool{thread_count};
            run_benchmark<int>(&bench, array_size, iterations, "HQarroum::thread_pool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   return pool.schedule(thread_task, std::cref(a), std::cref(b));
                               });
        }

        {
            std::cerr << "  oneTBB task_group\n";
            oneapi::tbb::global_control control{
                oneapi::tbb::global_control::max_allowed_parallelism, thread_count};
            oneapi::tbb::task_group group;
            run_benchmark<int>(&bench, array_size, iterations, "oneTBB task_group",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> std::future<void> {
                                   auto promise = std::make_shared<std::promise<void>>();
                                   auto future = promise->get_future();
                                   group.run([promise, &thread_task, &a, &b] {
                                       try {
                                           thread_task(a, b);
                                           promise->set_value();
                                       } catch (...) {
                                           promise->set_exception(std::current_exception());
                                       }
                                   });
                                   return future;
                               });
            group.wait();
        }

        std::cerr << "finished matrix multiplication " << array_size << "x" << array_size << "\n";
    }

    std::cout << "wrote " << RESULTS_MARKDOWN_FILE << '\n';
}
