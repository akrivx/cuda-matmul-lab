#pragma once

#include <string_view>

namespace cuda_matmul_lab {

enum class MatmulVersion {
    NAIVE = 0,       // One thread per output element, row-stride A and C accesses and B broadcasts
    NAIVE_COALESCED, // One thread per output element, coalesced B and C accesses
    CUBLAS,          // cuBLAS baseline
    COUNT
};

// Throws `std::invalid_argument` if `version` is not an implementation.
[[nodiscard]] std::string_view get_matmul_name(MatmulVersion version);

} // namespace cuda_matmul_lab
