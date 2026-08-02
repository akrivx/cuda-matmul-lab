cmake_minimum_required(VERSION 3.25)

project(
    cuda_matmul_lab
    VERSION 0.1.0
    DESCRIPTION "A staged CUDA SGEMM optimization study"
    LANGUAGES CXX CUDA
)

# For benchmark builds, select the local GPU explicitly with -DCMAKE_CUDA_ARCHITECTURES=native. It is intentionally not
# the default so that configuration also works on machines without a visible NVIDIA GPU.

# CUDA 13.0 supports CUDA C++ through C++20. Host-only sources can use C++23. Both defaults remain overridable from the
# CMake command line.
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# Kernel implementations and shared build requirements for every SGEMM stage. Link each executable to
# cuda_matmul_lab::cuda_matmul_lab.
add_library(
    cuda_matmul_lab
    STATIC
        src/backends/cublas.cpp
        src/backends/cublas.hpp
        src/backends/cublas_check.hpp
        src/benchmark.cpp
        src/cuda_check.hpp
        src/launch_topology.cpp
        src/matmul.cpp
        src/matmul_callback.hpp
        src/matmul_callback_factory.cpp
        src/matmul_callback_factory.hpp
        src/matmul_kernel_traits.cpp
        src/matmul_kernel_traits.hpp
        src/matmul_validation.cpp
        src/matmul_validation.hpp
        src/matrix_view.hpp
        src/resolved_launch_topology.hpp
        src/kernels/matmul_kernel_launchers.hpp
        src/kernels/naive.cu
        src/kernels/naive_coalesced.cu
)
add_library(cuda_matmul_lab::cuda_matmul_lab ALIAS cuda_matmul_lab)

# Give mixed C++/CUDA Visual Studio builds an explicit compiler PDB filename. Without it, some CMake/VS combinations use
# the intermediate directory itself as ProgramDataBaseFileName.
if(MSVC)
    set_target_properties(cuda_matmul_lab PROPERTIES COMPILE_PDB_NAME cuda_matmul_lab)
endif()

target_sources(
    cuda_matmul_lab
    PUBLIC
        FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            include/benchmark.hpp
            include/launch_topology.hpp
            include/matmul.hpp
)

target_include_directories(
    cuda_matmul_lab
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_features(
    cuda_matmul_lab
    PUBLIC
        cxx_std_23
        cuda_std_20
)

target_link_libraries(
    cuda_matmul_lab
    PUBLIC
        CUDA::cudart
    PRIVATE
        CUDA::cublas
)

# Drives BenchmarkSession to run and report on every implemented kernel stage.
add_executable(benchmark_runner apps/benchmark_runner.cpp)
target_link_libraries(benchmark_runner PRIVATE cuda_matmul_lab::cuda_matmul_lab)
