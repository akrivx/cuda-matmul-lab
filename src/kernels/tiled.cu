#include <algorithm>
#include <cstddef>

#include <cuda/cmath>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "launch_topology.hpp"
#include "matrix_view.hpp"

#include "kernels/matmul_kernel_launchers.hpp"

using cuda_matmul_lab::MatrixView;
using cuda_matmul_lab::TileShape;

namespace {

__device__ void load_tile(MatrixView<const float> matrix, std::size_t src_row, std::size_t src_col,
                          MatrixView<float> tile, unsigned dst_row, unsigned dst_col) {
    if (dst_row >= tile.extent(0) || dst_col >= tile.extent(1)) {
        return;
    }
    if (src_row < matrix.extent(0) && src_col < matrix.extent(1)) {
        tile(dst_row, dst_col) = matrix(src_row, src_col);
    } else {
        tile(dst_row, dst_col) = 0.0f;
    }
}

__global__ void tiled_matmul_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                    std::size_t tile_k_dim) {
    extern __shared__ float shared_mem[];

    auto A_tile = cuda_matmul_lab::make_matrix_view(shared_mem, blockDim.y, tile_k_dim);
    auto B_tile = cuda_matmul_lab::make_matrix_view(shared_mem + blockDim.y * tile_k_dim, tile_k_dim, blockDim.x);

    const std::size_t first_tile_row = std::size_t{blockIdx.y} * blockDim.y;
    const std::size_t first_tile_col = std::size_t{blockIdx.x} * blockDim.x;
    const std::size_t tile_row_stride = std::size_t{blockDim.y} * gridDim.y;
    const std::size_t tile_col_stride = std::size_t{blockDim.x} * gridDim.x;

    for (std::size_t tile_row = first_tile_row; tile_row < C.extent(0); tile_row += tile_row_stride) {
        for (std::size_t tile_col = first_tile_col; tile_col < C.extent(1); tile_col += tile_col_stride) {
            const std::size_t row = tile_row + threadIdx.y;
            const std::size_t col = tile_col + threadIdx.x;

            float value = 0.0f;
            for (std::size_t tile_k = 0; tile_k < A.extent(1); tile_k += tile_k_dim) {
                load_tile(A, row, tile_k + threadIdx.x, A_tile, threadIdx.y, threadIdx.x);
                load_tile(B, tile_k + threadIdx.y, col, B_tile, threadIdx.y, threadIdx.x);
                __syncthreads();

                for (std::size_t k = 0; k < tile_k_dim; ++k) {
                    value += A_tile(threadIdx.y, k) * B_tile(k, threadIdx.x);
                }
                __syncthreads();
            }

            if (row < C.extent(0) && col < C.extent(1)) {
                C(row, col) = value;
            }
        }
    }
}

} // namespace

namespace cuda_matmul_lab::detail {

void launch_tiled_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                         const ResolvedLaunchTopology& topology, cudaStream_t stream) {
    // With no explicit tile, derive one from the block so the constraints validate_and_resolve_topology enforces
    // (block.x == tile.n, block.y == tile.m, tile.n/m >= tile.k) hold automatically for whatever block was requested.
    const TileShape tile = topology.tile.value_or(TileShape{
        .m = topology.block.y,
        .n = topology.block.x,
        .k = std::min(topology.block.x, topology.block.y),
    });
    const std::size_t total_tile_bytes = (tile.m * tile.k + tile.k * tile.n) * sizeof(float);

    const auto required_blocks_x = cuda::ceil_div(C.extent(1), std::size_t{tile.n});
    const auto required_blocks_y = cuda::ceil_div(C.extent(0), std::size_t{tile.m});
    const auto grid_x = static_cast<unsigned>(std::min(required_blocks_x, std::size_t{topology.grid_cap.x}));
    const auto grid_y = static_cast<unsigned>(std::min(required_blocks_y, std::size_t{topology.grid_cap.y}));

    const dim3 block{topology.block.x, topology.block.y, 1};
    const dim3 grid{grid_x, grid_y, 1};
    tiled_matmul_kernel<<<grid, block, total_tile_bytes, stream>>>(A, B, C, tile.k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_matmul_lab::detail
