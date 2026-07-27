#pragma once

#include <cstdio>
#include <exception>

#include <cublas_v2.h>

namespace cuda_matmul_lab::detail {

inline void check_cublas_status(cublasStatus_t status, const char* expression, const char* file, int line) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "cuBLAS call failed at %s:%d\n"
                     "  expression: %s\n"
                     "  error: %s (%s, %d)\n",
                     file, line, expression, cublasGetStatusString(status), cublasGetStatusName(status),
                     static_cast<int>(status));
        std::terminate();
    }
}

} // namespace cuda_matmul_lab::detail

#define CUBLAS_CHECK(expression)                                                                                       \
    ::cuda_matmul_lab::detail::check_cublas_status((expression), #expression, __FILE__, __LINE__)
