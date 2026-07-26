#pragma once

#include <optional>

#include "matmul.hpp"
#include "matmul_detail.hpp"
#include "matrix_view.hpp"

namespace cuda_matmul_lab::detail {

void validate_matrix_shapes(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C);

// Validates the version-specific topology policy and CUDA device limits, then
// resolves an omitted grid cap. A library implementation returns std::nullopt.
[[nodiscard]] std::optional<ResolvedLaunchTopology>
validate_and_resolve_topology(MatmulVersion version, const std::optional<LaunchTopology>& topology);

} // namespace cuda_matmul_lab::detail
