#pragma once

#include <optional>

namespace cuda_matmul_lab {

// Dimensions of a CUDA thread block. Kernel factories must reject shapes that
// are unsupported by the selected implementation or the active CUDA device.
struct BlockShape {
    unsigned x;
    unsigned y;
    unsigned z = 1;
};

// Logical output and reduction tile dimensions used by tiled kernels. This is
// optional because simple kernels and library implementations do not expose a
// tile shape.
struct TileShape {
    unsigned m;
    unsigned n;
    unsigned k;
};

// Number of blocks to launch in each grid dimension. Used for capped launches,
// which require the kernel to cover the remaining work with grid-stride loops.
struct GridShape {
    unsigned x;
    unsigned y;
    unsigned z = 1;
};

// Launch settings for a hand-written CUDA kernel. Matrix dimensions are not
// part of the topology: the launcher derives the required grid from the output
// shape and clamps it to grid_cap (or the CUDA device limit when no cap is
// supplied).
struct LaunchTopology {
    BlockShape block;
    std::optional<TileShape> tile;
    std::optional<GridShape> grid_cap;
};

} // namespace cuda_matmul_lab
