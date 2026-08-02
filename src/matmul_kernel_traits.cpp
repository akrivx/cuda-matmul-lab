#include "matmul_kernel_traits.hpp"

#include <stdexcept>

#include "kernels/matmul_kernel_launchers.hpp"

namespace cuda_matmul_lab::detail {

MatmulKernelTraits get_matmul_kernel_traits(MatmulVersion version) {
    constexpr TopologyPolicy simple_kernel_policy{
        .topology = Requirement::REQUIRED,
        .tile = Requirement::FORBIDDEN,
        .grid_cap = Requirement::OPTIONAL,
    };

    switch (version) {
    case MatmulVersion::NAIVE:
        return {"naive", simple_kernel_policy, &launch_naive_kernel};

    case MatmulVersion::NAIVE_COALESCED:
        return {"naive_coalesced", simple_kernel_policy, &launch_naive_coalesced_kernel};

    case MatmulVersion::CUBLAS:
    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"matmul version is not backed by a generic kernel launcher"};
}

} // namespace cuda_matmul_lab::detail
