#pragma once

#include "matmul.hpp"
#include "resolved_launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Naive kernel: one thread computes one output element via a straight triple
// loop, with no coalescing, shared memory, or reuse.
MatmulFn get_naive_matmul(const ResolvedLaunchTopology& topology);

} // namespace cuda_matmul_lab::detail
