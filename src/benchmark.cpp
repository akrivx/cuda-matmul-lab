#include "benchmark.hpp"

#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuda_matmul_lab {

namespace {

[[nodiscard]] std::string make_benchmark_name(const BenchmarkCase& benchmark_case) {
    std::string name{get_matmul_name(benchmark_case.version)};
    if (benchmark_case.topology) {
        name += '_';
        name += to_string(*benchmark_case.topology);
    }
    return name;
}

struct UniqueCudaEvent {
    UniqueCudaEvent(const UniqueCudaEvent&) = delete;
    UniqueCudaEvent& operator=(const UniqueCudaEvent&) = delete;
    UniqueCudaEvent(UniqueCudaEvent&&) = delete;
    UniqueCudaEvent& operator=(UniqueCudaEvent&&) = delete;
    UniqueCudaEvent() { CUDA_CHECK(cudaEventCreate(&handle)); }
    ~UniqueCudaEvent() { CUDA_CHECK(cudaEventDestroy(handle)); }
    cudaEvent_t handle{};
};

[[nodiscard]] double compute_relative_error(double absolute_error, double reference_value) {
    if (reference_value == 0.0) {
        return (absolute_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    }
    return absolute_error / std::fabs(reference_value);
}

struct CorrectnessResult {
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool passed = true;
};

void validate_benchmark_inputs(const BenchmarkConfig& config, MatrixView<const float> host_reference,
                               MatrixView<float> device_result) {
    if (config.timed_iterations == 0) {
        throw std::invalid_argument{"timed_iterations must be positive"};
    }
    if (!std::isfinite(config.absolute_tolerance) || config.absolute_tolerance < 0.0) {
        throw std::invalid_argument{"absolute_tolerance must be finite and non-negative"};
    }
    if (!std::isfinite(config.relative_tolerance) || config.relative_tolerance < 0.0) {
        throw std::invalid_argument{"relative_tolerance must be finite and non-negative"};
    }
    if (host_reference.extent(0) != device_result.extent(0) || host_reference.extent(1) != device_result.extent(1)) {
        throw std::invalid_argument{"host reference dimensions must match the device result"};
    }
    if (host_reference.stride(1) != 1 || host_reference.stride(0) < host_reference.extent(1)) {
        throw std::invalid_argument{"host reference must use row-major layout with a valid row stride"};
    }
    if (host_reference.data_handle() == nullptr) {
        throw std::invalid_argument{"host reference must have a non-null data pointer"};
    }
}

[[nodiscard]] CorrectnessResult evaluate_correctness(const BenchmarkConfig& config,
                                                     MatrixView<const float> host_reference,
                                                     MatrixView<const float> host_result) {
    CorrectnessResult result;
    for (std::size_t row = 0; row < host_reference.extent(0); ++row) {
        for (std::size_t column = 0; column < host_reference.extent(1); ++column) {
            const double actual_value = static_cast<double>(host_result(row, column));
            const double reference_value = static_cast<double>(host_reference(row, column));

            if (!std::isfinite(actual_value) || !std::isfinite(reference_value)) {
                result.max_abs_error = std::numeric_limits<double>::infinity();
                result.max_rel_error = std::numeric_limits<double>::infinity();
                result.passed = false;
                return result;
            }

            const double abs_error = std::fabs(actual_value - reference_value);
            const double rel_error = compute_relative_error(abs_error, reference_value);
            result.max_abs_error = std::max(result.max_abs_error, abs_error);
            result.max_rel_error = std::max(result.max_rel_error, rel_error);

            const double tolerance = config.absolute_tolerance + config.relative_tolerance * std::fabs(reference_value);
            result.passed = result.passed && abs_error <= tolerance;
        }
    }
    return result;
}

[[nodiscard]] double compute_median_of_sorted(std::span<const double> values) {
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

[[nodiscard]] std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* error_message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error{error_message};
    }
    return lhs * rhs;
}

} // namespace

BenchmarkResult run_benchmark(const BenchmarkCase& benchmark_case, const BenchmarkConfig& config,
                              MatrixView<const float> host_reference, MatrixView<const float> A,
                              MatrixView<const float> B, MatrixView<float> C, cudaStream_t stream) {
    validate_benchmark_inputs(config, host_reference, C);
    MatmulFn matmul = get_matmul_callback(benchmark_case.version, benchmark_case.topology);

    for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
        matmul(A, B, C, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    UniqueCudaEvent start_event;
    UniqueCudaEvent stop_event;

    double total_ms = 0.0;
    std::vector<double> iteration_times_ms;
    iteration_times_ms.reserve(config.timed_iterations);

    for (std::size_t i = 0; i < config.timed_iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start_event.handle, stream));
        matmul(A, B, C, stream);
        CUDA_CHECK(cudaEventRecord(stop_event.handle, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event.handle));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event.handle, stop_event.handle));

        total_ms += static_cast<double>(elapsed_ms);
        iteration_times_ms.push_back(static_cast<double>(elapsed_ms));
    }

    std::sort(iteration_times_ms.begin(), iteration_times_ms.end());

    const double mean_ms = total_ms / static_cast<double>(config.timed_iterations);
    const double median_ms = compute_median_of_sorted(iteration_times_ms);
    const double min_ms = iteration_times_ms.front();

    const std::size_t m = A.extent(0);
    const std::size_t n = B.extent(1);
    const std::size_t k = B.extent(0);

    const double floating_point_operations =
        2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double gflops = floating_point_operations / (median_ms * 1.0e6);

    const std::size_t result_element_count = checked_product(m, n, "host result element count exceeds size_t");
    const std::size_t result_row_bytes = checked_product(n, sizeof(float), "host result row size exceeds size_t");
    const std::size_t device_result_pitch_bytes =
        checked_product(C.stride(0), sizeof(float), "device result row pitch exceeds size_t");
    auto host_result_storage = std::make_unique_for_overwrite<float[]>(result_element_count);
    CUDA_CHECK(cudaMemcpy2D(host_result_storage.get(), result_row_bytes, C.data_handle(), device_result_pitch_bytes,
                            result_row_bytes, m, cudaMemcpyDeviceToHost));

    const auto host_result = make_matrix_view(host_result_storage.get(), m, n);
    const CorrectnessResult correctness = evaluate_correctness(config, host_reference, host_result);
    return {
        .name = make_benchmark_name(benchmark_case),
        .topology = benchmark_case.topology,
        .mean_ms = mean_ms,
        .median_ms = median_ms,
        .min_ms = min_ms,
        .gflops = gflops,
        .max_abs_error = correctness.max_abs_error,
        .max_rel_error = correctness.max_rel_error,
        .correctness_passed = correctness.passed,
    };
}

} // namespace cuda_matmul_lab
