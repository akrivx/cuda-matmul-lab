#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "launch_topology.hpp"
#include "matmul.hpp"

namespace cuda_matmul_lab {

// Logical dimensions for A[M,K] * B[K,N] = C[M,N].
struct MatmulShape {
    std::size_t m = 1024;
    std::size_t n = 1024;
    std::size_t k = 1024;
};

// One implementation/topology combination to benchmark. Multiple cases may select the same version with different
// topologies. Reports derive each case's label from `version` and `topology`.
//
// cuBLAS uses a null topology; hand-written kernels require an explicit topology for reproducibility.
struct BenchmarkCase {
    MatmulVersion version;
    std::optional<LaunchTopology> topology;
};

// Timing and correctness metrics for one benchmark case. All time values are milliseconds, and `gflops` uses
// `median_ms`. `max_rel_error` is reference-relative and becomes infinity for nonzero error against a zero reference.
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

// Measurement, correctness, and deterministic input-generation settings shared by benchmark cases.
struct BenchmarkConfig {
    std::size_t warmup_iterations = 5;
    std::size_t timed_iterations = 5;

    // An element passes when:
    // abs(actual - reference) <= absolute_tolerance
    //                          + relative_tolerance * abs(reference).
    double absolute_tolerance = 1e-4;
    double relative_tolerance = 1e-3;
    std::uint64_t random_seed = 0xC0FFEE;
};

// Owns the backend resources shared by a sequence of benchmark runs. A session is bound to the CUDA device active
// during construction, is not safe for concurrent use, and must be destroyed while that device is current and before
// it is reset. Destruction releases the shared cuBLAS handle and therefore synchronizes that device.
class BenchmarkSession {
  public:
    BenchmarkSession();
    ~BenchmarkSession();

    BenchmarkSession(const BenchmarkSession&) = delete;
    BenchmarkSession& operator=(const BenchmarkSession&) = delete;

    // Allocates inputs and outputs, generates deterministic random inputs, computes a cuBLAS reference, and benchmarks
    // `cases` using `stream`. Returns only after all associated CUDA work and result transfers have completed.
    // Allocation, initialization, transfers, and callback construction are outside the timed intervals.
    //
    // `stream` must belong to the CUDA device to which this session is bound. Throws `std::invalid_argument` for
    // invalid configuration, shapes, or implementation settings, and `std::logic_error` if another device is active.
    [[nodiscard]] std::vector<BenchmarkResult> run(MatmulShape shape, std::span<const BenchmarkCase> cases,
                                                   const BenchmarkConfig& config, cudaStream_t stream);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Writes results as a Markdown table, adding block, tile, and grid-cap columns when topology information is available.
void write_report(std::ostream& out, std::string_view stage_title, MatmulShape shape, const BenchmarkConfig& config,
                  std::span<const BenchmarkResult> results);

} // namespace cuda_matmul_lab
