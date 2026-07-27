#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "matmul.hpp"

namespace cuda_matmul_lab {

struct MatmulShape {
    std::size_t m = 1024;
    std::size_t n = 1024;
    std::size_t k = 1024;
};

// One implementation/topology combination to benchmark. Multiple cases may select the same version with different
// topologies. `label`, when non-empty, distinguishes cases in reports; otherwise `get_matmul_name(version)` is used.
//
// cuBLAS uses a null topology; hand-written kernels require an explicit topology for reproducibility.
struct BenchmarkCase {
    MatmulVersion version;
    std::optional<LaunchTopology> topology;
    std::string label;
};

struct BenchmarkResult {
    std::string name;
    std::optional<LaunchTopology> topology;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double gflops = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool correctness_passed = false;
};

struct BenchmarkConfig {
    std::size_t warmup_iterations = 5;
    std::size_t timed_iterations = 5;

    // An element passes when:
    // abs(actual - reference) <= absolute_tolerance
    //                          + relative_tolerance * abs(reference).
    double absolute_tolerance = 1e-4;
    double relative_tolerance = 1e-3;
    std::uint64_t random_seed = 0xC0FFEE;

    // Cases are supplied explicitly by the application and may eventually be loaded from a configuration file.
    // Hand-written kernels require a topology; library implementations such as cuBLAS require `std::nullopt`.
    std::vector<BenchmarkCase> cases;
};

// Enqueues warm-up and timed invocations of a matmul callback on `stream`, measures the timed interval with CUDA events
// recorded on that stream, and validates the final output against `host_reference`.
//
// `A` and `B` refer to device-accessible input matrices, `C` refers to a device-accessible output matrix, and
// `host_reference` refers to host-accessible memory. Their extents must describe `A[M,K] * B[K,N] = C[M,N]`.
//
// `gflops` is the conventional `2 * M * N * K` operation count divided by `median_ms`.
[[nodiscard]] BenchmarkResult run_benchmark(const BenchmarkCase& benchmark_case, const BenchmarkConfig& config,
                                            MatrixView<const float> host_reference, MatrixView<const float> A,
                                            MatrixView<const float> B, MatrixView<float> C, cudaStream_t stream);

// Allocates inputs and outputs, generates deterministic random inputs, computes a reference, and benchmarks every case
// in `config.cases`. Allocation, initialization, transfers, and callback construction are outside the timed intervals.
[[nodiscard]] std::vector<BenchmarkResult> run_all_benchmarks(MatmulShape shape, const BenchmarkConfig& config);

// Writes results as a Markdown table, adding block, tile, and grid-cap columns when topology information is available.
void write_report(std::ostream& out, std::string_view stage_title, MatmulShape shape, const BenchmarkConfig& config,
                  std::span<const BenchmarkResult> results);

} // namespace cuda_matmul_lab
