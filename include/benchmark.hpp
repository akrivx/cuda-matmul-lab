#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
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

struct BenchmarkResult {
    std::string name;
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
};

// Enqueues warm-up and timed invocations on `stream`, measures the timed
// interval with CUDA events recorded on that stream, and validates the final
// output against `host_reference`.
//
// A and B refer to device-accessible input matrices, C refers to a
// device-accessible output matrix, and host_reference refers to host-accessible
// memory. Their extents must describe A[M,K] * B[K,N] = C[M,N].
//
// `gflops` is the conventional 2*M*N*K operation count divided by median_ms.
[[nodiscard]] BenchmarkResult
run_benchmark(std::string_view name, const MatmulFn& matmul,
              const BenchmarkConfig& config,
              MatrixView<const float> host_reference, MatrixView<const float> A,
              MatrixView<const float> B, MatrixView<float> C,
              cudaStream_t stream);

// Allocates inputs and outputs, generates deterministic random input matrices,
// computes a reference result, then benchmarks every registered implementation.
// Allocation, initialization, transfers, and callback construction are not
// included in the timed intervals.
[[nodiscard]] std::vector<BenchmarkResult>
run_all_benchmarks(MatmulShape shape, const BenchmarkConfig& config);

// Writes a benchmark run's results as a Markdown table.
void write_report(std::ostream& out, std::string_view stage_title,
                  MatmulShape shape, const BenchmarkConfig& config,
                  const std::vector<BenchmarkResult>& results);

} // namespace cuda_matmul_lab
