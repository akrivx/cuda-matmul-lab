#pragma once

#include <optional>

#include "matmul.hpp"

namespace cuda_matmul_lab::detail {

// Device-specific form consumed by kernel factories. Unlike LaunchTopology,
// grid_cap has been resolved to an effective CUDA grid limit.
struct ResolvedLaunchTopology {
    BlockShape block;
    GridShape grid_cap;
    std::optional<TileShape> tile;
};

// Naive kernel: one thread computes one output element via a straight triple
// loop, with no coalescing, shared memory, or reuse.
MatmulFn get_naive_matmul(const ResolvedLaunchTopology& topology);

// cuBLAS reference used as the correctness and performance target.
MatmulFn get_reference_matmul();

} // namespace cuda_matmul_lab::detail
