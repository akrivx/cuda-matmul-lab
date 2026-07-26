#include "matmul.hpp"

#include <stdexcept>
#include <string_view>

namespace cuda_matmul_lab {

namespace {

// Classifies implementation versions and rejects sentinels or invalid values.
[[nodiscard]] bool needs_topology(MatmulVersion version) {
    switch (version) {
    case MatmulVersion::NAIVE:
        return true;
    case MatmulVersion::CUBLAS:
        return false;
    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

} // namespace

[[nodiscard]] std::string_view get_matmul_name(MatmulVersion version) {
    switch (version) {
    case MatmulVersion::NAIVE:
        return "naive";
    case MatmulVersion::CUBLAS:
        return "cublas-reference";
    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

[[nodiscard]] MatmulFn get_matmul_callback(MatmulVersion version, const std::optional<LaunchTopology>& topology) {
    if (needs_topology(version) != topology.has_value()) {
        throw std::invalid_argument{"matmul topology does not match implementation requirements"};
    }

    switch (version) {
    case MatmulVersion::NAIVE:
        return detail::get_naive_matmul(*topology);
    case MatmulVersion::CUBLAS:
        return detail::get_reference_matmul();
    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

} // namespace cuda_matmul_lab
