#pragma once

#include <functional>
#include <optional>
#include <string_view>

#include <cuda_runtime_api.h>

#include "launch_topology.hpp"
#include "matrix_view.hpp"

namespace cuda_matmul_lab {

enum class MatmulVersion {
    NAIVE = 0, // One thread per output element
    CUBLAS,    // cuBLAS reference
    COUNT
};

// Common signature shared by every implementation under study.
// Enqueues C = A * B on stream, where A is MxK, B is KxN, and C is MxN.
// Every view refers to device memory and must remain valid until the queued
// work completes. A callback may own mutable library state and is not required
// to support concurrent invocations.
using MatmulFn =
    std::function<void(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C, cudaStream_t stream)>;

// Throws std::invalid_argument if version is not an implementation.
[[nodiscard]] std::string_view get_matmul_name(MatmulVersion version);

// Creates a callback for `version`. topology must contain a value exactly when
// that implementation requires one: hand-written kernels require a topology,
// while implementations such as cuBLAS require std::nullopt.
// The topology is validated and resolved against the active CUDA device once;
// invoke the returned callback on that same device.
// Throws std::invalid_argument for an invalid version or topology.
[[nodiscard]] MatmulFn get_matmul_callback(MatmulVersion version, const std::optional<LaunchTopology>& topology);

} // namespace cuda_matmul_lab
