#pragma once

#include "matmul.hpp"
#include "resolved_launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Naive kernel: one thread computes one output element without coalescing, shared memory, or data reuse.
MatmulFn get_naive_matmul(const ResolvedLaunchTopology& topology);

} // namespace cuda_matmul_lab::detail
