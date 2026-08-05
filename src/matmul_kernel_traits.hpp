#pragma once

#include <string_view>

#include <cuda_runtime_api.h>

#include "matmul.hpp"
#include "matrix_view.hpp"
#include "resolved_launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Whether a LaunchTopology field is disallowed, allowed either way, or mandatory for a given matmul version.
enum class Requirement {
    FORBIDDEN,
    OPTIONAL,
    REQUIRED,
};

// Per-version rules for which LaunchTopology fields topology validation and resolution accepts.
struct TopologyPolicy {
    Requirement topology;
    Requirement tile;
};

// Launches an already-validated, resolved kernel invocation. Matches every hand-written kernel launcher's signature.
using LaunchFn = void (*)(MatrixView<const float>, MatrixView<const float>, MatrixView<float>,
                          const ResolvedLaunchTopology&, cudaStream_t);

// Static facts about one matmul version backed by a generic, stateless kernel launcher.
struct MatmulKernelTraits {
    std::string_view name;
    TopologyPolicy policy;
    LaunchFn launch;
};

// Throws `std::invalid_argument` if `version` is not backed by a generic kernel launcher, either because it isn't
// an implementation at all, or because (like cuBLAS) it's backed by a stateful library handle that
// MatmulCallbackFactory constructs separately. Callers that must also support those versions special-case them
// before falling back to this table; see get_matmul_name and validate_and_resolve_topology.
[[nodiscard]] MatmulKernelTraits get_matmul_kernel_traits(MatmulVersion version);

} // namespace cuda_matmul_lab::detail
