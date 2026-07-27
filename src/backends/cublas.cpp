#include "backends/cublas.hpp"

#include <limits>

#include <cublas_v2.h>

#include "cublas_check.hpp"
#include "matmul_validation.hpp"

namespace cuda_matmul_lab::detail {

struct CublasBackend::State {
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State() { CUBLAS_CHECK(cublasCreate(&handle)); }
    ~State() { CUBLAS_CHECK(cublasDestroy(handle)); }
    cublasHandle_t handle{};
};

CublasBackend::CublasBackend() : state_{std::make_shared<State>()} {}

MatmulFn CublasBackend::make_callback() const {
    return [state = state_](MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                            cudaStream_t stream) {
        constexpr std::size_t cublas_int_max = std::numeric_limits<int>::max();
        validate_matrix_shapes(A, B, C, cublas_int_max, cublas_int_max);

        const int M = static_cast<int>(A.extent(0));
        const int N = static_cast<int>(B.extent(1));
        const int K = static_cast<int>(B.extent(0));
        const int lda = static_cast<int>(B.stride(0));
        const int ldb = static_cast<int>(A.stride(0));
        const int ldc = static_cast<int>(C.stride(0));
        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(state->handle, stream));

        // A row-major MxK matrix is a column-major KxM matrix over the same storage. Compute C^T = B^T * A^T so cuBLAS
        // writes row-major C, preserving each view's padded row stride as the leading dimension.
        CUBLAS_CHECK(cublasSgemm(state->handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data_handle(), lda,
                                 A.data_handle(), ldb, &beta, C.data_handle(), ldc));
    };
}

} // namespace cuda_matmul_lab::detail
