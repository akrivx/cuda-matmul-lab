#include "matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include <cuda/cmath>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

using cuda_matmul_lab::GridShape;
using cuda_matmul_lab::LaunchTopology;
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

void validate_matrix_shapes(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C) {
    if (A.extent(0) != C.extent(0) || B.extent(1) != C.extent(1) || A.extent(1) != B.extent(0)) {
        std::ostringstream out;
        out << "naive matmul kernel: invalid matrix dimensions (A=" << A.extent(0) << 'x' << A.extent(1)
            << ", B=" << B.extent(0) << 'x' << B.extent(1) << ", C=" << C.extent(0) << 'x' << C.extent(1) << ')';
        throw std::invalid_argument{out.str()};
    }
}

[[nodiscard]] GridShape validate_topology(const LaunchTopology& topology) {
    if (topology.tile) {
        throw std::invalid_argument{"naive matmul kernel: does not accept tile topology"};
    }

    if (topology.block.x == 0 || topology.block.y == 0 || topology.block.z != 1) {
        throw std::invalid_argument{"naive matmul kernel: block dimensions x and y must be positive and z must be 1"};
    }

    if (topology.grid_cap && (topology.grid_cap->x == 0 || topology.grid_cap->y == 0 || topology.grid_cap->z != 1)) {
        throw std::invalid_argument{"naive matmul kernel: grid cap x and y must be positive and z must be 1"};
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::uint64_t threads_per_block = std::uint64_t{topology.block.x} * topology.block.y * topology.block.z;
    if (topology.block.x > static_cast<unsigned>(properties.maxThreadsDim[0]) ||
        topology.block.y > static_cast<unsigned>(properties.maxThreadsDim[1]) ||
        topology.block.z > static_cast<unsigned>(properties.maxThreadsDim[2]) ||
        threads_per_block > static_cast<std::uint64_t>(properties.maxThreadsPerBlock)) {
        throw std::invalid_argument{"naive matmul kernel: block shape exceeds the active CUDA device limits"};
    }

    const GridShape device_grid_limit{
        static_cast<unsigned>(properties.maxGridSize[0]),
        static_cast<unsigned>(properties.maxGridSize[1]),
        static_cast<unsigned>(properties.maxGridSize[2]),
    };
    const GridShape grid_limit = topology.grid_cap.value_or(device_grid_limit);
    if (grid_limit.x > device_grid_limit.x || grid_limit.y > device_grid_limit.y ||
        grid_limit.z > device_grid_limit.z) {
        throw std::invalid_argument{"naive matmul kernel: grid cap exceeds the active CUDA device limits"};
    }

    return grid_limit;
}

} // namespace

namespace cuda_matmul_lab::detail {

MatmulFn get_naive_matmul(const LaunchTopology& topology) {
    const GridShape grid_limit = validate_topology(topology);

    return [topology, grid_limit](MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                  cudaStream_t stream) {
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
        const auto grid_x = static_cast<unsigned>(std::min(required_blocks_x, std::size_t{grid_limit.x}));
        const auto grid_y = static_cast<unsigned>(std::min(required_blocks_y, std::size_t{grid_limit.y}));

        const dim3 block{topology.block.x, topology.block.y, 1};
        const dim3 grid{grid_x, grid_y, 1};
        naive_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C);
        CUDA_CHECK(cudaGetLastError());
    };
}

} // namespace cuda_matmul_lab::detail
