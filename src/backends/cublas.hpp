#pragma once

#include <memory>

#include "matmul_callback.hpp"

namespace cuda_matmul_lab::detail {

// Owns one cuBLAS handle and creates callbacks that share its lifetime. The callbacks are intended for sequential use
// because setting a stream mutates handle state.
class CublasBackend {
  public:
    CublasBackend();

    CublasBackend(const CublasBackend&) = delete;
    CublasBackend& operator=(const CublasBackend&) = delete;

    [[nodiscard]] MatmulFn make_callback() const;

  private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace cuda_matmul_lab::detail
