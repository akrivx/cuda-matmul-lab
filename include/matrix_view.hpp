#pragma once

#include <cassert>
#include <cstddef>

#include <cuda/std/array>
#include <cuda/std/mdspan>

namespace cuda_matmul_lab {

// A non-owning row-major view. The leading dimension is the stride, in
// elements, between successive rows and may include row padding. The pointer's
// memory space is determined by the caller.
template <typename Element>
using MatrixView =
    cuda::std::mdspan<Element, cuda::std::dextents<std::size_t, 2>,
                      cuda::std::layout_stride>;

template <typename Element>
[[nodiscard]] constexpr MatrixView<Element>
make_matrix_view(Element* data, std::size_t rows, std::size_t columns,
                 std::size_t leading_dimension) noexcept {
    assert(leading_dimension > 0);
    assert(leading_dimension >= columns);

    using View = MatrixView<Element>;
    return View{data,
                typename View::mapping_type{
                    typename View::extents_type{rows, columns},
                    cuda::std::array<std::size_t, 2>{leading_dimension, 1}}};
}

template <typename Element>
[[nodiscard]] constexpr MatrixView<Element>
make_matrix_view(Element* data, std::size_t rows,
                 std::size_t columns) noexcept {
    return make_matrix_view(data, rows, columns, columns == 0 ? 1 : columns);
}

} // namespace cuda_matmul_lab
