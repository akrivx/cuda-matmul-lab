#include <cstddef>

#include <cuda/cmath>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/matmul_kernel_launchers.hpp"

using cuda_matmul_lab::MatrixView;

namespace {

__global__ void naive_coalesced_matmul_kernel(MatrixView<const float> A, MatrixView<const float> B,
                                              MatrixView<float> C) {
    // This baseline maps adjacent x threads to adjacent columns, producing coalesced B and C accesses (col varies
    // contiguously across the warp) and A broadcasts (each warp thread accesses the same address of A).
    const std::size_t row = std::size_t{blockIdx.y} * blockDim.y + threadIdx.y;
    const std::size_t col = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;

    if (row >= C.extent(0) || col >= C.extent(1)) {
        return;
    }

    float value = 0.0f;
    for (std::size_t k = 0; k < A.extent(1); ++k) {
        value += A(row, k) * B(k, col);
    }
    C(row, col) = value;
}

} // namespace

namespace cuda_matmul_lab::detail {

void launch_naive_coalesced_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                   const ResolvedLaunchTopology& topology, cudaStream_t stream) {
    // CUDA y covers matrix rows.
    const auto grid_x = static_cast<unsigned>(cuda::ceil_div(C.extent(1), std::size_t{topology.block.x}));
    const auto grid_y = static_cast<unsigned>(cuda::ceil_div(C.extent(0), std::size_t{topology.block.y}));

    const dim3 block{topology.block.x, topology.block.y};
    const dim3 grid{grid_x, grid_y};
    naive_coalesced_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_matmul_lab::detail
