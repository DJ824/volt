#include <nanobench.h>
#include <thread_pool/thread_pool.h>

#include <BS_thread_pool_light.hpp>
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
    requires std::invocable<FutureProvider, std::vector<DataType>, std::vector<DataType>>
void run_benchmark(ankerl::nanobench::Bench* bench, const std::size_t array_size,
                   const std::size_t multiplications_to_perform, const char* name,
                   FutureProvider&& provider) {
    // generate test data
    // const auto computations = generate_benchmark_data<int>(array_size,
    // multiplications_to_perform);
    const std::vector<DataType> a(array_size, 2);
    const std::vector<DataType> b(array_size, 3);

    // set up vector for results
    std::vector<std::future<void>> results;

    if constexpr (!std::is_void_v<std::invoke_result_t<FutureProvider, std::vector<DataType>,
                                                       std::vector<DataType>>>) {
        results.reserve(multiplications_to_perform);
    }

    bench->run(name, [&]() {
        for (std::size_t i = 0; i < multiplications_to_perform; ++i) {
            // let std async decide on how to launch the task, either deferred or async
            if constexpr (std::is_same_v<std::future<void>,
                                         std::invoke_result_t<FutureProvider, std::vector<DataType>,
                                                              std::vector<DataType>>>) {
                results.emplace_back(provider(std::ref(a), std::ref(b)));

            } else {
                provider(std::ref(a), std::ref(b));
            }
        }

        if (!results.empty()) {
            // wait for futures
            for (auto& fut : results) {
                fut.wait();
            }
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
                [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                    pool.submit_detached(thread_task, a, b);
                });
        }

        {
            std::cerr << "  dp::thread_pool - std::function\n";
            dp::thread_pool<std::function<void()>> pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations, "dp::thread_pool - std::function",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                                   pool.enqueue_detach(thread_task, a, b);
                               });
        }

#ifdef __cpp_lib_move_only_function
        {
            std::cerr << "  dp::thread_pool - std::move_only_function\n";
            dp::thread_pool pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations,
                               "dp::thread_pool - std::move_only_function",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                                   pool.enqueue_detach(thread_task, a, b);
                               });
        }
#endif
        {
            std::cerr << "  dp::thread_pool - fu2::unique_function\n";
            dp::thread_pool<fu2::unique_function<void()>> pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations,
                               "dp::thread_pool - fu2::unique_function",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                                   pool.enqueue_detach(thread_task, a, b);
                               });
        }

        {
            std::cerr << "  BS::thread_pool\n";
            BS::thread_pool_light bs_thread_pool{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations, "BS::thread_pool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                                   bs_thread_pool.push_task(thread_task, a, b);
                               });
        }

        {
            std::cerr << "  task_thread_pool\n";
            task_thread_pool::task_thread_pool ttp{static_cast<unsigned int>(thread_count)};
            run_benchmark<int>(&bench, array_size, iterations, "task_thread_pool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                                   ttp.submit_detach(thread_task, a, b);
                               });
        }

        {
            std::cerr << "  riften::Thiefpool\n";
            riften::Thiefpool riften_thiefpool{thread_count};
            run_benchmark<int>(&bench, array_size, iterations, "riften::Thiefpool",
                               [&](const std::vector<int>& a, const std::vector<int>& b) -> void {
                                   riften_thiefpool.enqueue_detach(thread_task, a, b);
                               });
        }

        std::cerr << "finished matrix multiplication " << array_size << "x" << array_size << "\n";
    }

    std::cout << "wrote " << RESULTS_MARKDOWN_FILE << '\n';
}
