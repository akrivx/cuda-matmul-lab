#include "matmul_callback_factory.hpp"

#include "kernels/matmul_kernel_launchers.hpp"
#include "matmul_validation.hpp"

#include <stdexcept>

namespace cuda_matmul_lab::detail {

MatmulFn MatmulCallbackFactory::make(MatmulVersion version, const std::optional<LaunchTopology>& topology) {
    const auto resolved_topology = validate_and_resolve_topology(version, topology);

    switch (version) {
    case MatmulVersion::NAIVE:
        return [topology = *resolved_topology](MatrixView<const float> A, MatrixView<const float> B,
                                               MatrixView<float> C, cudaStream_t stream) {
            validate_matrix_shapes(A, B, C);
            launch_naive_kernel(A, B, C, topology, stream);
        };

    case MatmulVersion::CUBLAS:
        if (!cublas_backend_) {
            cublas_backend_.emplace();
        }
        return cublas_backend_->make_callback();

    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

} // namespace cuda_matmul_lab::detail
