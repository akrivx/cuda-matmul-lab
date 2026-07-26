#pragma once

#include <optional>

#include "launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Device-specific form consumed by kernel factories. Unlike LaunchTopology,
// grid_cap has been resolved to an effective CUDA grid limit.
struct ResolvedLaunchTopology {
    BlockShape block;
    GridShape grid_cap;
    std::optional<TileShape> tile;
};

} // namespace cuda_matmul_lab::detail
