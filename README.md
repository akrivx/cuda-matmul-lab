# cuda-matmul-lab

A staged study of optimizing SGEMM (single-precision matrix multiplication)
in CUDA, starting from a naive one-thread-per-element kernel and working up
towards cuBLAS-competitive performance. Each stage changes one thing,
gets profiled with Nsight Compute, and gets a short written report.

## Roadmap

| Stage | Name                              | Focus |
|-------|------------------------------------|-------|
| 1     | Naive                              | Correctness baseline, one thread per output element |
| 2     | Coalesced                          | Fix global memory access pattern |
| 3     | Tiled                              | Shared memory tiling / data reuse |
| 4/5   | Thread-tiled                       | Per-thread register blocking |
| 6     | Warp-tiled                         | Warp-level tiling + `__shfl_sync` |
| 7     | Vectorized                         | `float4` memory access |
| 8     | Bank-conflict-free                 | Shared memory layout / padding |
| 9     | Double-buffered                    | Software pipelining (prefetch next tile) |
| 10    | cuBLAS                             | Reference / ground truth, not optimized by hand |
| —     | Tensor Cores (WMMA)                | Optional bonus stage, different programming model |

Each stage should be a single, isolated change from the previous one so
that the Nsight Compute delta is attributable to it.
