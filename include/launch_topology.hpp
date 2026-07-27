#pragma once

#include <optional>
#include <string>

namespace cuda_matmul_lab {

// CUDA thread-block dimensions, validated by kernel factories against the implementation and active device.
struct BlockShape {
    unsigned x;
    unsigned y;
    unsigned z = 1;
};

// Optional logical output and reduction tile dimensions exposed by tiled kernels.
struct TileShape {
    unsigned m;
    unsigned n;
    unsigned k;
};

// Maximum blocks to launch in each grid dimension; grid-stride loops must cover work beyond this cap.
struct GridShape {
    unsigned x;
    unsigned y;
    unsigned z = 1;
};

// Requested launch settings for a hand-written CUDA kernel. An omitted `grid_cap` uses the active device's grid limit.
struct LaunchTopology {
    BlockShape block;
    std::optional<GridShape> grid_cap;
    std::optional<TileShape> tile;
};

[[nodiscard]] std::string to_string(const LaunchTopology& topology);

} // namespace cuda_matmul_lab
