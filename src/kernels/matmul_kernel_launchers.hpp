#pragma once

#include <cuda_runtime_api.h>

#include "matrix_view.hpp"
#include "resolved_launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Launches the deliberately uncoalesced correctness baseline.
void launch_naive_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                         const ResolvedLaunchTopology& topology, cudaStream_t stream);

// Launches the naive CUDA matmul with coalesced B and C accesses.
void launch_naive_coalesced_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                   const ResolvedLaunchTopology& topology, cudaStream_t stream);

void launch_tiled_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                         const ResolvedLaunchTopology& topology, cudaStream_t stream);

void launch_thread_tiled_kernel(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                const ResolvedLaunchTopology& topology, cudaStream_t stream);

} // namespace cuda_matmul_lab::detail
