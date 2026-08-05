#pragma once

#include <optional>
#include <string>

namespace cuda_matmul_lab {

// CUDA thread-block dimensions, validated by kernel factories against the implementation and active device. Every
// kernel in this project launches 2D grids/blocks, so there is no z dimension to track.
struct BlockShape {
    unsigned x;
    unsigned y;
};

// Optional logical output and reduction tile dimensions exposed by tiled kernels.
struct TileShape {
    unsigned m;
    unsigned n;
    unsigned k;
};

// Requested launch settings for a hand-written CUDA kernel.
struct LaunchTopology {
    BlockShape block;
    std::optional<TileShape> tile;
};

[[nodiscard]] std::string to_string(const LaunchTopology& topology);

} // namespace cuda_matmul_lab
