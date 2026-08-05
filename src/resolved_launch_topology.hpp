#pragma once

#include <optional>

#include "launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Device-specific topology consumed by kernel factories, with `grid_cap` resolved to an effective CUDA grid limit.
struct ResolvedLaunchTopology {
    BlockShape block;
    std::optional<TileShape> tile;
    GridShape grid_cap;
};

} // namespace cuda_matmul_lab::detail
