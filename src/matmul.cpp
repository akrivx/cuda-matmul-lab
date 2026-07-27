#include "matmul.hpp"

#include <stdexcept>

namespace cuda_matmul_lab {

std::string_view get_matmul_name(MatmulVersion version) {
    switch (version) {
    case MatmulVersion::NAIVE:
        return "naive";
    case MatmulVersion::CUBLAS:
        return "cublas";
    default:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

} // namespace cuda_matmul_lab
