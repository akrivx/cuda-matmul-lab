#include "launch_topology.hpp"

#include <format>

namespace cuda_matmul_lab {

std::string to_string(const LaunchTopology& topology) {
    std::string result = std::format("block{}x{}", topology.block.x, topology.block.y);
    if (topology.tile) {
        result += std::format("_tile{}x{}x{}", topology.tile->m, topology.tile->n, topology.tile->k);
    }
    if (topology.grid_cap) {
        result += std::format("_grid_cap{}x{}", topology.grid_cap->x, topology.grid_cap->y);
    }
    return result;
}

} // namespace cuda_matmul_lab
