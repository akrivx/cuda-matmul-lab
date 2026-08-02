#include "matmul.hpp"

#include <stdexcept>

namespace cuda_matmul_lab {

std::string_view get_matmul_name(MatmulVersion version) {
    switch (version) {
    case MatmulVersion::NAIVE:
        return "naive";
    case MatmulVersion::NAIVE_COALESCED:
        return "naive_coalesced";
    case MatmulVersion::CUBLAS:
        return "cublas";
    case MatmulVersion::COUNT:
        break;
    }

    throw std::invalid_argument{"unknown matmul version"};
}

} // namespace cuda_matmul_lab
