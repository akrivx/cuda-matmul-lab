#pragma once

#include <cstddef>
#include <optional>

#include "matmul.hpp"
#include "matrix_view.hpp"
#include "resolved_launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Validates compatible, non-empty row-major views. A zero limit means
// unlimited; otherwise extents and row strides are checked independently.
void validate_matrix_shapes(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                            std::size_t max_extent = 0, std::size_t max_stride = 0);

// Validates the version-specific topology policy and CUDA device limits, then
// resolves an omitted grid cap. A library implementation returns std::nullopt.
[[nodiscard]] std::optional<ResolvedLaunchTopology>
validate_and_resolve_topology(MatmulVersion version, const std::optional<LaunchTopology>& topology);

} // namespace cuda_matmul_lab::detail
