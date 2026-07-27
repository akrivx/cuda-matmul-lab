#pragma once

#include <optional>

#include "backends/cublas.hpp"
#include "launch_topology.hpp"
#include "matmul.hpp"
#include "matmul_callback.hpp"

namespace cuda_matmul_lab::detail {

// Owns backend resources and creates validated callbacks for every selectable matrix-multiplication implementation.
// Callbacks created by one factory are intended for sequential use on the CUDA device active at construction.
class MatmulCallbackFactory {
  public:
    // Validates `topology` against the selected implementation and active device.
    // Throws `std::invalid_argument` for an invalid version or topology.
    [[nodiscard]] MatmulFn make(MatmulVersion version, const std::optional<LaunchTopology>& topology);

  private:
    std::optional<CublasBackend> cublas_backend_;
};

} // namespace cuda_matmul_lab::detail
