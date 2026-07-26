#include "matmul_factories.hpp"

#include "matmul_validation.hpp"

#include "kernels/matmul_kernel_launchers.hpp"

namespace cuda_matmul_lab::detail {

MatmulFn get_naive_matmul(const ResolvedLaunchTopology& topology) {
    return [topology](MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C, cudaStream_t stream) {
        validate_matrix_shapes(A, B, C);
        launch_naive_kernel(A, B, C, topology, stream);
    };
}

} // namespace cuda_matmul_lab::detail
