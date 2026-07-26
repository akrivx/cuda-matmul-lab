#pragma once

#include "matmul.hpp"

namespace cuda_matmul_lab::detail {

// cuBLAS reference used as the correctness and performance target.
MatmulFn get_cublas_matmul();

} // namespace cuda_matmul_lab::detail
