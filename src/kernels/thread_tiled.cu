#include <cstddef>
#include <stdexcept>

#include <cuda/cmath>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "launch_topology.hpp"
#include "matrix_view.hpp"

#include "kernels/matmul_kernel_launchers.hpp"

using cuda_matmul_lab::MatrixView;
using cuda_matmul_lab::TileShape;

namespace {

// Cooperatively fills `tile` from `matrix`, with `tile`'s (0,0) corresponding to matrix(global_row, global_col).
// Every thread in the block loads several elements (striding by the block's thread count) so this works regardless
// of how `tile` compares in size to the block's thread count. Out-of-bounds source reads are zero-padded.
__device__ void load_shared_tile(MatrixView<const float> matrix, MatrixView<float> tile, std::size_t global_row,
                                 std::size_t global_col) {
    const unsigned threads_per_block = blockDim.x * blockDim.y;
    const unsigned linear_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const auto tile_cols = static_cast<unsigned>(tile.extent(1));
    const auto tile_elements = static_cast<unsigned>(tile.extent(0)) * tile_cols;

    for (unsigned idx = linear_thread_id; idx < tile_elements; idx += threads_per_block) {
        const unsigned local_row = idx / tile_cols;
        const unsigned local_col = idx % tile_cols;
        const std::size_t src_row = global_row + local_row;
        const std::size_t src_col = global_col + local_col;

        tile(local_row, local_col) =
            (src_row < matrix.extent(0) && src_col < matrix.extent(1)) ? matrix(src_row, src_col) : 0.0f;
    }
}

template <unsigned TM, unsigned TN>
__global__ void thread_tiled_matmul_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                           TileShape block_tile) {
    extern __shared__ float shared_mem[];

    auto A_tile = cuda_matmul_lab::make_matrix_view(shared_mem, block_tile.m, block_tile.k);
    auto B_tile =
        cuda_matmul_lab::make_matrix_view(shared_mem + block_tile.m * block_tile.k, block_tile.k, block_tile.n);

    const std::size_t tile_row = std::size_t{blockIdx.y} * block_tile.m;
    const std::size_t tile_col = std::size_t{blockIdx.x} * block_tile.n;

    float A_reg[TM];
    float B_reg[TN];
    float accum_reg[TM][TN] = {0.0f};

    for (std::size_t tile_k = 0; tile_k < A.extent(1); tile_k += block_tile.k) {
        load_shared_tile(A, A_tile, tile_row, tile_k);
        load_shared_tile(B, B_tile, tile_k, tile_col);
        __syncthreads();

        for (std::size_t k = 0; k < block_tile.k; ++k) {
            for (unsigned y = 0; y < TM; ++y) {
                A_reg[y] = A_tile(threadIdx.y * TM + y, k);
            }
            for (unsigned x = 0; x < TN; ++x) {
                B_reg[x] = B_tile(k, threadIdx.x * TN + x);
            }
            for (unsigned y = 0; y < TM; ++y) {
                for (unsigned x = 0; x < TN; ++x) {
                    accum_reg[y][x] += A_reg[y] * B_reg[x];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned y = 0; y < TM; ++y) {
        for (unsigned x = 0; x < TN; ++x) {
            const auto out_row = tile_row + threadIdx.y * TM + y;
            const auto out_col = tile_col + threadIdx.x * TN + x;
            if (out_row < C.extent(0) && out_col < C.extent(1)) {
                C(out_row, out_col) = accum_reg[y][x];
            }
        }
    }
}

[[noreturn]] void throw_invalid_thread_tile_shape() {
    throw std::invalid_argument{"thread tiled kernel expects 4, 8, or 16 for any of tile.m/n/k."};
}

template <unsigned TN>
void launch_thread_tiled_matmul_kernel_impl_dynamic_tm(MatrixView<const float> A, MatrixView<const float> B,
                                                       MatrixView<float> C, TileShape block_tile,
                                                       unsigned thread_tile_m, dim3 grid, dim3 block,
                                                       std::size_t total_tile_bytes, cudaStream_t stream) {
    if (thread_tile_m == 4u) {
        thread_tiled_matmul_kernel<4u, TN><<<grid, block, total_tile_bytes, stream>>>(A, B, C, block_tile);
    } else if (thread_tile_m == 8u) {
        thread_tiled_matmul_kernel<8u, TN><<<grid, block, total_tile_bytes, stream>>>(A, B, C, block_tile);
    } else if (thread_tile_m == 16u) {
        thread_tiled_matmul_kernel<16u, TN><<<grid, block, total_tile_bytes, stream>>>(A, B, C, block_tile);
    } else {
        // Unreachable in practice: validate_and_resolve_topology already rejects any other tile.m. Kept as a clear
        // failure instead of silently doing nothing if that invariant is ever broken.
        throw_invalid_thread_tile_shape();
    }
}

void launch_thread_tiled_matmul_kernel_impl(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                            TileShape block_tile, TileShape thread_tile, dim3 grid, dim3 block,
                                            std::size_t total_tile_bytes, cudaStream_t stream) {
    if (thread_tile.n == 4u) {
        launch_thread_tiled_matmul_kernel_impl_dynamic_tm<4u>(A, B, C, block_tile, thread_tile.m, grid, block,
                                                              total_tile_bytes, stream);
    } else if (thread_tile.n == 8u) {
        launch_thread_tiled_matmul_kernel_impl_dynamic_tm<8u>(A, B, C, block_tile, thread_tile.m, grid, block,
                                                              total_tile_bytes, stream);
    } else if (thread_tile.n == 16u) {
        launch_thread_tiled_matmul_kernel_impl_dynamic_tm<16u>(A, B, C, block_tile, thread_tile.m, grid, block,
                                                               total_tile_bytes, stream);
    } else {
        // Unreachable in practice; see the note above.
        throw_invalid_thread_tile_shape();
    }
}

} // namespace

namespace cuda_matmul_lab::detail {

void launch_thread_tiled_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                const ResolvedLaunchTopology& topology, cudaStream_t stream) {
    const TileShape thread_tile = topology.tile.value();
    const TileShape block_tile = {topology.block.y * thread_tile.m, topology.block.x * thread_tile.n, thread_tile.k};

    const std::size_t total_tile_bytes = (block_tile.m * block_tile.k + block_tile.k * block_tile.n) * sizeof(float);

    const auto grid_x = static_cast<unsigned>(cuda::ceil_div(C.extent(1), std::size_t{block_tile.n}));
    const auto grid_y = static_cast<unsigned>(cuda::ceil_div(C.extent(0), std::size_t{block_tile.m}));

    const dim3 block{topology.block.x, topology.block.y};
    const dim3 grid{grid_x, grid_y};
    launch_thread_tiled_matmul_kernel_impl(A, B, C, block_tile, thread_tile, grid, block, total_tile_bytes, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_matmul_lab::detail
