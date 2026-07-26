#include "matmul.hpp"

#include "backends/cublas.hpp"
#include "matmul_factories.hpp"
#include "matmul_validation.hpp"

#include <stdexcept>

namespace cuda_matmul_lab {

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
    auto resolved_topology = detail::validate_and_resolve_topology(version, topology);

    switch (version) {
    case MatmulVersion::NAIVE:
        return detail::get_naive_matmul(*resolved_topology);
    case MatmulVersion::CUBLAS:
        return detail::get_cublas_matmul();
    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

} // namespace cuda_matmul_lab
