#include "launch_topology.hpp"

#include <format>

namespace cuda_matmul_lab {

std::string to_string(const LaunchTopology& topology) {
    std::string result = std::format("block{}x{}", topology.block.x, topology.block.y);
    if (topology.tile) {
        result += std::format("_tile{}x{}x{}", topology.tile->m, topology.tile->n, topology.tile->k);
    }
    return result;
}

} // namespace cuda_matmul_lab
