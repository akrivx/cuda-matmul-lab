#pragma once

#include <optional>

#include "launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Device-specific topology consumed by kernel factories.
struct ResolvedLaunchTopology {
    BlockShape block;
    std::optional<TileShape> tile;
};

} // namespace cuda_matmul_lab::detail
