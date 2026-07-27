#pragma once

#include <cstdio>
#include <exception>

#include <cuda_runtime.h>

namespace cuda_matmul_lab::detail {

inline void check_cuda_error(cudaError_t status, const char* expression, const char* file, int line) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA call failed at %s:%d\n"
                     "  expression: %s\n"
                     "  error: %s (%d)\n",
                     file, line, expression, cudaGetErrorString(status), static_cast<int>(status));
        std::terminate();
    }
}

} // namespace cuda_matmul_lab::detail

#define CUDA_CHECK(expression)                                                                                         \
    ::cuda_matmul_lab::detail::check_cuda_error((expression), #expression, __FILE__, __LINE__)
