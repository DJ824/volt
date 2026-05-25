cmake_minimum_required(VERSION 3.20)
project(volt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VOLT_BENCH_FETCH_DEPS "Fetch nanobench and comparison pool dependencies" ON)
option(VOLT_BENCH_NATIVE_ARCH "Build benchmarks with -march=native/-mtune=native" ON)
option(VOLT_BENCH_LTO "Enable interprocedural optimization/LTO for benchmarks" ON)

find_package(Threads REQUIRED)

set(VOLT_RUNTIME_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

add_executable(volt main.cpp)
set_target_properties(
    volt
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${VOLT_RUNTIME_OUTPUT_DIR}"
)
target_include_directories(volt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(thread_pool_bench bench/thread_pool_bench.cpp
        include/task_free_list.h)
set_target_properties(
    thread_pool_bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${VOLT_RUNTIME_OUTPUT_DIR}"
)
target_include_directories(
    thread_pool_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/benchmark/include"
)
target_link_libraries(thread_pool_bench PRIVATE Threads::Threads)
target_compile_definitions(
    thread_pool_bench
    PRIVATE
        NDEBUG
        ANKERL_NANOBENCH_LOG_ENABLED=0
        RESULTS_MARKDOWN_FILE="${CMAKE_CURRENT_BINARY_DIR}/thread_pool_benchmark_results.md"
)

include(CheckIPOSupported)
check_ipo_supported(RESULT VOLT_BENCH_IPO_SUPPORTED OUTPUT VOLT_BENCH_IPO_ERROR)
if (VOLT_BENCH_LTO AND VOLT_BENCH_IPO_SUPPORTED)
    set_property(TARGET thread_pool_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(VOLT_BENCH_LTO)
    message(WARNING "IPO/LTO requested but unavailable: ${VOLT_BENCH_IPO_ERROR}")
endif()

target_compile_options(
    thread_pool_bench
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
            -O3
            -DNDEBUG
            -Wall
            -Wextra
            -fomit-frame-pointer
            -fno-semantic-interposition
        >
        $<$<AND:$<BOOL:${VOLT_BENCH_NATIVE_ARCH}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:
            -march=native
            -mtune=native
        >
        $<$<CXX_COMPILER_ID:MSVC>:
            /O2
            /Ob3
            /DNDEBUG
        >
)

target_link_options(
    thread_pool_bench
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
            -fno-semantic-interposition
        >
        $<$<CXX_COMPILER_ID:MSVC>:
            /OPT:REF
            /OPT:ICF
        >
)

if (VOLT_BENCH_FETCH_DEPS)
    include(FetchContent)

    FetchContent_Declare(
        nanobench
        GIT_REPOSITORY https://github.com/martinus/nanobench.git
        GIT_TAG v4.3.11
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nanobench)
    target_link_libraries(thread_pool_bench PRIVATE nanobench)

    FetchContent_Declare(
        bs_thread_pool
        GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
        GIT_TAG v3.5.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(bs_thread_pool)
    target_include_directories(thread_pool_bench PRIVATE "${bs_thread_pool_SOURCE_DIR}/include")

    FetchContent_Declare(
        task_thread_pool
        GIT_REPOSITORY https://github.com/alugowski/task-thread-pool.git
        GIT_TAG v1.0.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(task_thread_pool)
    if (TARGET task-thread-pool::task-thread-pool)
        target_link_libraries(thread_pool_bench PRIVATE task-thread-pool::task-thread-pool)
    else()
        target_include_directories(thread_pool_bench PRIVATE "${task_thread_pool_SOURCE_DIR}/include")
    endif()

    FetchContent_Declare(
        riften_threadpool
        GIT_REPOSITORY https://github.com/ConorWilliams/Threadpool.git
        GIT_TAG a757265d75c189ad0fd0e429c166efb6281f2770
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(riften_threadpool)
    if (TARGET RiftenThiefpool)
        target_link_libraries(thread_pool_bench PRIVATE RiftenThiefpool)
    endif()
    target_include_directories(
        thread_pool_bench
        PRIVATE
            "${riften_threadpool_SOURCE_DIR}/include"
            "${CMAKE_BINARY_DIR}/_deps/function2-src/include"
            "${CMAKE_BINARY_DIR}/_deps/concurrentdeque-src/include"
    )
else()
    find_package(nanobench REQUIRED)
    target_link_libraries(thread_pool_bench PRIVATE nanobench)
endif()
