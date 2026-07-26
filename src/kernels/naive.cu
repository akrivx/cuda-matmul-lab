#include <algorithm>
#include <cstddef>
#include <utility>

#include <cuda/cmath>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "matmul_factories.hpp"
#include "matmul_validation.hpp"

using cuda_matmul_lab::MatrixView;

namespace {

__global__ void naive_matmul_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C) {
    // This baseline intentionally maps adjacent x threads to different rows.
    // Consequently, warps access columns with a row stride rather than using
    // the coalesced x-to-column mapping introduced by a later implementation.
    const std::size_t first_row = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t first_col = std::size_t{blockIdx.y} * blockDim.y + threadIdx.y;
    const std::size_t row_stride = std::size_t{blockDim.x} * gridDim.x;
    const std::size_t col_stride = std::size_t{blockDim.y} * gridDim.y;

    for (std::size_t row = first_row; row < C.extent(0); row += row_stride) {
        for (std::size_t col = first_col; col < C.extent(1); col += col_stride) {
            float value = 0.0f;
            for (std::size_t k = 0; k < A.extent(1); ++k) {
                value += A(row, k) * B(k, col);
            }
            C(row, col) = value;
        }
    }
}

} // namespace

namespace cuda_matmul_lab::detail {

MatmulFn get_naive_matmul(const ResolvedLaunchTopology& topology) {
    return [topology](MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C, cudaStream_t stream) {
        validate_matrix_shapes(A, B, C);

        // CUDA does not permit a zero-sized grid. An output with no elements is
        // already complete and requires no launch.
        if (C.extent(0) == 0 || C.extent(1) == 0) {
            return;
        }

        // CUDA x deliberately covers matrix rows in this baseline. The grid
        // may be capped; the kernel's grid-stride loops cover any remainder.
        const auto required_blocks_x = cuda::ceil_div(C.extent(0), std::size_t{topology.block.x});
        const auto required_blocks_y = cuda::ceil_div(C.extent(1), std::size_t{topology.block.y});
        const auto grid_x = static_cast<unsigned>(std::min(required_blocks_x, std::size_t{topology.grid_cap.x}));
        const auto grid_y = static_cast<unsigned>(std::min(required_blocks_y, std::size_t{topology.grid_cap.y}));

        const dim3 block{topology.block.x, topology.block.y, 1};
        const dim3 grid{grid_x, grid_y, 1};
        naive_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C);
        CUDA_CHECK(cudaGetLastError());
    };
}

} // namespace cuda_matmul_lab::detail
