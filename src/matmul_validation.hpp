#pragma once

#include <cstddef>
#include <optional>

#include "launch_topology.hpp"
#include "matmul.hpp"
#include "matrix_view.hpp"
#include "resolved_launch_topology.hpp"

namespace cuda_matmul_lab::detail {

// Validates that the logical M, N, and K matrix dimensions are positive.
void validate_matmul_dimensions(std::size_t M, std::size_t N, std::size_t K);

// Validates compatible, non-empty row-major views, with independent optional limits for extents and row strides. A
// zero limit means unbounded.
void validate_matrix_shapes(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                            std::size_t max_extent = 0, std::size_t max_stride = 0);

// Validates topology policy and device limits, then resolves an omitted grid cap; library backends return
// `std::nullopt`.
[[nodiscard]] std::optional<ResolvedLaunchTopology>
validate_and_resolve_topology(MatmulVersion version, const std::optional<LaunchTopology>& topology);

} // namespace cuda_matmul_lab::detail
