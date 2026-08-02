#include "matmul.hpp"

#include "matmul_kernel_traits.hpp"

namespace cuda_matmul_lab {

std::string_view get_matmul_name(MatmulVersion version) {
    if (version == MatmulVersion::CUBLAS) {
        return "cublas";
    }
    return detail::get_matmul_kernel_traits(version).name;
}

} // namespace cuda_matmul_lab
