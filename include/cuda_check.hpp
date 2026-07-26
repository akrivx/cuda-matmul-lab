#pragma once

#include <cstdio>
#include <exception>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace cuda_matmul_lab::detail {

inline void check_cuda_error(cudaError_t status, const char* expression,
                             const char* file, int line) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA call failed at %s:%d\n"
                     "  expression: %s\n"
                     "  error: %s (%d)\n",
                     file, line, expression, cudaGetErrorString(status),
                     static_cast<int>(status));
        std::terminate();
    }
}

inline void check_cublas_status(cublasStatus_t status, const char* expression,
                                const char* file, int line) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "cuBLAS call failed at %s:%d\n"
                     "  expression: %s\n"
                     "  error: %s (%s, %d)\n",
                     file, line, expression, cublasGetStatusString(status),
                     cublasGetStatusName(status), static_cast<int>(status));
        std::terminate();
    }
}

} // namespace cuda_matmul_lab::detail

#define CUDA_CHECK(expression)                                                 \
    ::cuda_matmul_lab::detail::check_cuda_error((expression), #expression,     \
                                                __FILE__, __LINE__)

#define CUBLAS_CHECK(expression)                                               \
    ::cuda_matmul_lab::detail::check_cublas_status((expression), #expression,  \
                                                   __FILE__, __LINE__)
