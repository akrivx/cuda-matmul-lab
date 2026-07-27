#include "benchmark.hpp"

#include "cuda_check.hpp"
#include "matmul_validation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
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

// Fills a host-accessible matrix with values uniformly distributed in [-1, 1).
void fill_random_matrix(MatrixView<float> matrix, std::mt19937_64& generator) {
    std::uniform_real_distribution<float> distribution{-1.0f, 1.0f};

    for (std::size_t row = 0; row < matrix.extent(0); ++row) {
        for (std::size_t column = 0; column < matrix.extent(1); ++column) {
            matrix(row, column) = distribution(generator);
        }
    }
}

void copy_matrix_async(MatrixView<const float> src, MatrixView<float> dst, cudaMemcpyKind kind, cudaStream_t stream) {
    if (src.extent(0) != dst.extent(0) || src.extent(1) != dst.extent(1)) {
        throw std::invalid_argument{"src and dst matrix dimensions must match"};
    }

    const std::size_t src_pitch_bytes =
        checked_product(src.stride(0), sizeof(float), "src matrix row pitch exceeds size_t");
    const std::size_t dst_pitch_bytes =
        checked_product(dst.stride(0), sizeof(float), "dst matrix row pitch exceeds size_t");
    const std::size_t row_size_bytes = checked_product(src.extent(1), sizeof(float), "matrix row size exceeds size_t");

    CUDA_CHECK(cudaMemcpy2DAsync(dst.data_handle(), dst_pitch_bytes, src.data_handle(), src_pitch_bytes, row_size_bytes,
                                 src.extent(0), kind, stream));
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

class UniqueCudaMatrixStorage {
  public:
    UniqueCudaMatrixStorage(const UniqueCudaMatrixStorage&) = delete;
    UniqueCudaMatrixStorage& operator=(const UniqueCudaMatrixStorage&) = delete;
    UniqueCudaMatrixStorage(UniqueCudaMatrixStorage&&) = delete;
    UniqueCudaMatrixStorage& operator=(UniqueCudaMatrixStorage&&) = delete;

    UniqueCudaMatrixStorage(std::size_t rows, std::size_t columns) : rows_{rows}, columns_{columns} {
        const std::size_t row_size_bytes = checked_product(columns, sizeof(float), "matrix row size exceeds size_t");

        void* data = nullptr;
        CUDA_CHECK(cudaMallocPitch(&data, &pitch_bytes_, row_size_bytes, rows));

        data_.reset(static_cast<float*>(data));

        if (pitch_bytes_ % sizeof(float) != 0) {
            throw std::runtime_error{"matrix row pitch must be divisible by sizeof(float)"};
        }
    }

    [[nodiscard]] auto view() noexcept { return make_matrix_view(data_.get(), rows_, columns_, leading_dimension()); }

    [[nodiscard]] auto const_view() const noexcept {
        return make_matrix_view<const float>(data_.get(), rows_, columns_, leading_dimension());
    }

  private:
    struct Deleter {
        void operator()(float* ptr) const noexcept { CUDA_CHECK(cudaFree(ptr)); }
    };

    [[nodiscard]] std::size_t leading_dimension() const noexcept { return pitch_bytes_ / sizeof(float); }

    std::unique_ptr<float, Deleter> data_;
    std::size_t rows_;
    std::size_t columns_;
    std::size_t pitch_bytes_{};
};

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

std::vector<BenchmarkResult> run_all_benchmarks(MatmulShape shape, std::span<const BenchmarkCase> cases,
                                                const BenchmarkConfig& config, cudaStream_t stream) {
    detail::validate_matmul_dimensions(shape.m, shape.n, shape.k);

    const std::size_t a_element_count = checked_product(shape.m, shape.k, "A element count exceeds size_t");
    const std::size_t b_element_count = checked_product(shape.k, shape.n, "B element count exceeds size_t");
    const std::size_t c_element_count = checked_product(shape.m, shape.n, "C element count exceeds size_t");

    auto host_a_storage = std::make_unique_for_overwrite<float[]>(a_element_count);
    auto host_b_storage = std::make_unique_for_overwrite<float[]>(b_element_count);
    auto host_reference_storage = std::make_unique_for_overwrite<float[]>(c_element_count);

    UniqueCudaMatrixStorage device_a_storage{shape.m, shape.k};
    UniqueCudaMatrixStorage device_b_storage{shape.k, shape.n};
    UniqueCudaMatrixStorage device_c_storage{shape.m, shape.n};

    {
        std::mt19937_64 generator{config.random_seed};
        auto host_a = make_matrix_view(host_a_storage.get(), shape.m, shape.k);
        auto host_b = make_matrix_view(host_b_storage.get(), shape.k, shape.n);
        fill_random_matrix(host_a, generator);
        fill_random_matrix(host_b, generator);
        copy_matrix_async(host_a, device_a_storage.view(), cudaMemcpyHostToDevice, stream);
        copy_matrix_async(host_b, device_b_storage.view(), cudaMemcpyHostToDevice, stream);
    }

    {
        auto host_reference_output = make_matrix_view(host_reference_storage.get(), shape.m, shape.n);
        auto reference_matmul = get_matmul_callback(MatmulVersion::CUBLAS, std::nullopt);
        reference_matmul(device_a_storage.const_view(), device_b_storage.const_view(), device_c_storage.view(), stream);
        copy_matrix_async(device_c_storage.const_view(), host_reference_output, cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const auto host_reference = make_matrix_view<const float>(host_reference_storage.get(), shape.m, shape.n);
    const auto device_a = device_a_storage.const_view();
    const auto device_b = device_b_storage.const_view();
    auto device_c = device_c_storage.view();
    const std::size_t device_c_pitch_bytes =
        checked_product(device_c.stride(0), sizeof(float), "C row pitch exceeds size_t");
    const std::size_t device_c_row_size_bytes =
        checked_product(device_c.extent(1), sizeof(float), "C row size exceeds size_t");

    std::vector<BenchmarkResult> results;
    results.reserve(cases.size());
    for (const auto& benchmark_case : cases) {
        CUDA_CHECK(cudaMemset2DAsync(device_c.data_handle(), device_c_pitch_bytes, 0, device_c_row_size_bytes,
                                     device_c.extent(0), stream));
        results.push_back(run_benchmark(benchmark_case, config, host_reference, device_a, device_b, device_c, stream));
    }

    return results;
}

} // namespace cuda_matmul_lab
