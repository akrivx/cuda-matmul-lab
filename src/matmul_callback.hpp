#pragma once

#include <functional>

#include <cuda_runtime_api.h>

#include "matrix_view.hpp"

namespace cuda_matmul_lab::detail {

// Common internal signature for implementations under study. Enqueues `C = A * B` on `stream`; the device-accessible
// matrix views must remain valid until the queued work completes. Callbacks are not required to support concurrency.
using MatmulFn =
    std::function<void(MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C, cudaStream_t stream)>;

} // namespace cuda_matmul_lab::detail
