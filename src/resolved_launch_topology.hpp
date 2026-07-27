#pragma once

#include <optional>

#include "launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Device-specific topology consumed by kernel factories, with `grid_cap` resolved to an effective CUDA grid limit.
struct ResolvedLaunchTopology {
    BlockShape block;
    GridShape grid_cap;
    std::optional<TileShape> tile;
};

} // namespace cuda_matmul_lab::detail
