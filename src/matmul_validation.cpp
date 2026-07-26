#include "matmul_validation.hpp"

#include <cstdint>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace cuda_matmul_lab::detail {

namespace {

enum class Requirement {
    FORBIDDEN,
    OPTIONAL,
    REQUIRED,
};

struct TopologyPolicy {
    Requirement topology;
    Requirement tile;
    Requirement grid_cap;
};

[[nodiscard]] TopologyPolicy get_topology_policy(MatmulVersion version) {
    switch (version) {
    case MatmulVersion::NAIVE:
        return {
            .topology = Requirement::REQUIRED,
            .tile = Requirement::FORBIDDEN,
            .grid_cap = Requirement::OPTIONAL,
        };

    case MatmulVersion::CUBLAS:
        return {
            .topology = Requirement::FORBIDDEN,
            .tile = Requirement::FORBIDDEN,
            .grid_cap = Requirement::FORBIDDEN,
        };

    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

[[nodiscard]] constexpr bool satisfies(Requirement requirement, bool present) noexcept {
    switch (requirement) {
    case Requirement::FORBIDDEN:
        return !present;
    case Requirement::OPTIONAL:
        return true;
    case Requirement::REQUIRED:
        return present;
    }

    return false;
}

} // namespace

void validate_matrix_shapes(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                            std::size_t max_extent, std::size_t max_stride) {
    if (A.extent(0) != C.extent(0) || B.extent(1) != C.extent(1) || A.extent(1) != B.extent(0)) {
        std::ostringstream out;
        out << "invalid matrix dimensions (A=" << A.extent(0) << 'x' << A.extent(1) << ", B=" << B.extent(0) << 'x'
            << B.extent(1) << ", C=" << C.extent(0) << 'x' << C.extent(1) << ')';
        throw std::invalid_argument{out.str()};
    }

    const std::size_t M = C.extent(0);
    const std::size_t N = C.extent(1);
    const std::size_t K = A.extent(1);
    if (M == 0 || N == 0 || K == 0) {
        std::ostringstream out;
        out << "matrix dimensions M, N, and K must be positive (M=" << M << ", N=" << N << ", K=" << K << ')';
        throw std::invalid_argument{out.str()};
    }

    if (A.stride(1) != 1 || B.stride(1) != 1 || C.stride(1) != 1 || A.stride(0) < A.extent(1) ||
        B.stride(0) < B.extent(1) || C.stride(0) < C.extent(1)) {
        throw std::invalid_argument{"matrix views must use row-major layout with valid row strides"};
    }

    if (A.data_handle() == nullptr || B.data_handle() == nullptr || C.data_handle() == nullptr) {
        throw std::invalid_argument{"non-empty matrix views must have non-null data pointers"};
    }

    if (max_extent != 0 && (M > max_extent || N > max_extent || K > max_extent)) {
        std::ostringstream out;
        out << "matrix dimensions exceed max extent (max_extent=" << max_extent << ", M=" << M << ", N=" << N
            << ", K=" << K << ')';
        throw std::invalid_argument{out.str()};
    }

    if (max_stride != 0 &&
        (A.stride(0) > max_stride || B.stride(0) > max_stride || C.stride(0) > max_stride)) {
        std::ostringstream out;
        out << "matrix row strides exceed max stride (max_stride=" << max_stride << ", A=" << A.stride(0)
            << ", B=" << B.stride(0) << ", C=" << C.stride(0) << ')';
        throw std::invalid_argument{out.str()};
    }
}

[[nodiscard]] std::optional<ResolvedLaunchTopology>
validate_and_resolve_topology(MatmulVersion version, const std::optional<LaunchTopology>& topology) {
    const auto policy = get_topology_policy(version);

    if (!satisfies(policy.topology, topology.has_value())) {
        throw std::invalid_argument{"matmul topology does not match implementation requirements"};
    }

    if (!topology) {
        return std::nullopt;
    }

    if (!satisfies(policy.tile, topology->tile.has_value())) {
        throw std::invalid_argument{"matmul tile topology does not match implementation requirements"};
    }
    if (!satisfies(policy.grid_cap, topology->grid_cap.has_value())) {
        throw std::invalid_argument{"matmul grid-cap topology does not match implementation requirements"};
    }

    if (topology->block.x == 0 || topology->block.y == 0 || topology->block.z != 1) {
        throw std::invalid_argument{"block dimensions x and y must be positive and z must be 1"};
    }

    if (topology->grid_cap &&
        (topology->grid_cap->x == 0 || topology->grid_cap->y == 0 || topology->grid_cap->z != 1)) {
        throw std::invalid_argument{"grid cap x and y must be positive and z must be 1"};
    }

    if (topology->tile && (topology->tile->m == 0 || topology->tile->n == 0 || topology->tile->k == 0)) {
        throw std::invalid_argument{"tile dimensions m, n, and k must be positive"};
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::uint64_t threads_per_block = std::uint64_t{topology->block.x} * topology->block.y * topology->block.z;
    if (topology->block.x > static_cast<unsigned>(properties.maxThreadsDim[0]) ||
        topology->block.y > static_cast<unsigned>(properties.maxThreadsDim[1]) ||
        topology->block.z > static_cast<unsigned>(properties.maxThreadsDim[2]) ||
        threads_per_block > static_cast<std::uint64_t>(properties.maxThreadsPerBlock)) {
        throw std::invalid_argument{"block shape exceeds the active CUDA device limits"};
    }

    const GridShape device_grid_limit{
        static_cast<unsigned>(properties.maxGridSize[0]),
        static_cast<unsigned>(properties.maxGridSize[1]),
        static_cast<unsigned>(properties.maxGridSize[2]),
    };
    const GridShape grid_limit = topology->grid_cap.value_or(device_grid_limit);
    if (grid_limit.x > device_grid_limit.x || grid_limit.y > device_grid_limit.y ||
        grid_limit.z > device_grid_limit.z) {
        throw std::invalid_argument{"grid cap exceeds the active CUDA device limits"};
    }

    return ResolvedLaunchTopology{
        .block = topology->block,
        .grid_cap = grid_limit,
        .tile = topology->tile,
    };
}

} // namespace cuda_matmul_lab::detail
