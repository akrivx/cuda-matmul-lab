#include "matmul_callback_factory.hpp"

#include "matmul_kernel_traits.hpp"
#include "matmul_validation.hpp"

#include <stdexcept>

namespace cuda_matmul_lab::detail {

MatmulFn MatmulCallbackFactory::make(MatmulVersion version, const std::optional<LaunchTopology>& topology) {
    const auto resolved_topology = validate_and_resolve_topology(version, topology);

    if (version == MatmulVersion::CUBLAS) {
        if (!cublas_backend_) {
            cublas_backend_.emplace();
        }
        return cublas_backend_->make_callback();
    }

    const auto launch = get_matmul_kernel_traits(version).launch;

    return [topology = *resolved_topology, launch](MatrixView<const float> A, MatrixView<const float> B,
                                                   MatrixView<float> C, cudaStream_t stream) {
        validate_matrix_shapes(A, B, C);
        launch(A, B, C, topology, stream);
    };
}

} // namespace cuda_matmul_lab::detail
