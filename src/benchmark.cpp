#include "benchmark.hpp"

#include "cuda_check.hpp"
#include "matmul_callback.hpp"
#include "matmul_callback_factory.hpp"
#include "matmul_validation.hpp"
#include "matrix_view.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <ostream>
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

void validate_benchmark_config(const BenchmarkConfig& config) {
    if (config.timed_iterations == 0) {
        throw std::invalid_argument{"timed_iterations must be positive"};
    }
    if (!std::isfinite(config.absolute_tolerance) || config.absolute_tolerance < 0.0) {
        throw std::invalid_argument{"absolute_tolerance must be finite and non-negative"};
    }
    if (!std::isfinite(config.relative_tolerance) || config.relative_tolerance < 0.0) {
        throw std::invalid_argument{"relative_tolerance must be finite and non-negative"};
    }
}

void validate_benchmark_matrices(MatrixView<const double> host_reference, MatrixView<float> device_result) {
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
                                                     MatrixView<const double> host_reference,
                                                     MatrixView<const float> host_result) {
    CorrectnessResult result;
    for (std::size_t row = 0; row < host_reference.extent(0); ++row) {
        for (std::size_t column = 0; column < host_reference.extent(1); ++column) {
            const double actual_value = static_cast<double>(host_result(row, column));
            const double reference_value = host_reference(row, column);

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

// Computes `C = A * B` in increasing reduction-index order using double-precision accumulation.
void compute_cpu_reference(MatrixView<const float> A, MatrixView<const float> B, MatrixView<double> C) {
    for (std::size_t row = 0; row < C.extent(0); ++row) {
        for (std::size_t column = 0; column < C.extent(1); ++column) {
            C(row, column) = 0.0;
        }
    }

    for (std::size_t row = 0; row < C.extent(0); ++row) {
        for (std::size_t inner = 0; inner < A.extent(1); ++inner) {
            const double a = static_cast<double>(A(row, inner));
            for (std::size_t column = 0; column < C.extent(1); ++column) {
                C(row, column) += a * static_cast<double>(B(inner, column));
            }
        }
    }
}

void copy_matrix_async(MatrixView<const float> source, MatrixView<float> destination, cudaMemcpyKind kind,
                       cudaStream_t stream) {
    if (source.extent(0) != destination.extent(0) || source.extent(1) != destination.extent(1)) {
        throw std::invalid_argument{"source and destination matrix dimensions must match"};
    }

    const std::size_t source_pitch_bytes =
        checked_product(source.stride(0), sizeof(float), "source matrix row pitch exceeds size_t");
    const std::size_t destination_pitch_bytes =
        checked_product(destination.stride(0), sizeof(float), "destination matrix row pitch exceeds size_t");
    const std::size_t row_size_bytes =
        checked_product(source.extent(1), sizeof(float), "matrix row size exceeds size_t");

    CUDA_CHECK(cudaMemcpy2DAsync(destination.data_handle(), destination_pitch_bytes, source.data_handle(),
                                 source_pitch_bytes, row_size_bytes, source.extent(0), kind, stream));
}

// RAII owner for one CUDA timing event.
struct CudaEvent {
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&handle)); }
    ~CudaEvent() { CUDA_CHECK(cudaEventDestroy(handle)); }
    cudaEvent_t handle{};
};

// Owns a pitched device allocation for one float matrix. Views remain valid until the storage is destroyed.
class PitchedDeviceMatrixStorage {
  public:
    PitchedDeviceMatrixStorage(std::size_t rows, std::size_t columns)
        : rows_{rows}, columns_{columns},
          row_size_bytes_{checked_product(columns, sizeof(float), "matrix row size exceeds size_t")} {
        void* data = nullptr;
        CUDA_CHECK(cudaMallocPitch(&data, &pitch_bytes_, row_size_bytes_, rows));

        data_.reset(static_cast<float*>(data));

        if (pitch_bytes_ % sizeof(float) != 0) {
            throw std::runtime_error{"matrix row pitch must be divisible by sizeof(float)"};
        }
    }

    [[nodiscard]] auto view() noexcept { return make_matrix_view(data_.get(), rows_, columns_, leading_dimension()); }

    [[nodiscard]] auto const_view() const noexcept {
        return make_matrix_view<const float>(data_.get(), rows_, columns_, leading_dimension());
    }

    // Enqueues zeroing of the logical matrix elements on `stream`; row padding is left unchanged.
    void zero_async(cudaStream_t stream) noexcept {
        CUDA_CHECK(cudaMemset2DAsync(data_.get(), pitch_bytes_, 0, row_size_bytes_, rows_, stream));
    }

  private:
    struct Deleter {
        void operator()(float* ptr) const noexcept { CUDA_CHECK(cudaFree(ptr)); }
    };

    [[nodiscard]] std::size_t leading_dimension() const noexcept { return pitch_bytes_ / sizeof(float); }

    std::unique_ptr<float, Deleter> data_;
    std::size_t rows_;
    std::size_t columns_;
    std::size_t row_size_bytes_;
    std::size_t pitch_bytes_{};
};

// Measures one already-constructed callback and compares its final result with the host reference.
[[nodiscard]] BenchmarkResult run_case(const BenchmarkCase& benchmark_case, const BenchmarkConfig& config,
                                       const detail::MatmulFn& matmul, MatrixView<const double> host_reference,
                                       MatrixView<const float> A, MatrixView<const float> B, MatrixView<float> C,
                                       cudaStream_t stream) {
    for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
        matmul(A, B, C, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaEvent start_event;
    CudaEvent stop_event;

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

[[nodiscard]] int get_current_cuda_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

} // namespace

struct BenchmarkSession::Impl {
    void validate_current_device() const {
        if (get_current_cuda_device() != device) {
            throw std::logic_error{"another CUDA device is active; BenchmarkSession is bound to its creation device"};
        }
    }

    int device{get_current_cuda_device()};
    detail::MatmulCallbackFactory callback_factory;
};

BenchmarkSession::BenchmarkSession() : impl_{std::make_unique<Impl>()} {}

BenchmarkSession::~BenchmarkSession() = default;

std::vector<BenchmarkResult> BenchmarkSession::run(MatmulShape shape, std::span<const BenchmarkCase> cases,
                                                   const BenchmarkConfig& config, cudaStream_t stream) {
    impl_->validate_current_device();
    validate_benchmark_config(config);
    detail::validate_matmul_dimensions(shape.m, shape.n, shape.k);

    if (cases.empty()) {
        return {};
    }

    const std::size_t a_element_count = checked_product(shape.m, shape.k, "A element count exceeds size_t");
    const std::size_t b_element_count = checked_product(shape.k, shape.n, "B element count exceeds size_t");
    const std::size_t c_element_count = checked_product(shape.m, shape.n, "C element count exceeds size_t");

    auto host_a_storage = std::make_unique_for_overwrite<float[]>(a_element_count);
    auto host_b_storage = std::make_unique_for_overwrite<float[]>(b_element_count);
    auto host_reference_storage = std::make_unique_for_overwrite<double[]>(c_element_count);

    PitchedDeviceMatrixStorage device_a_storage{shape.m, shape.k};
    PitchedDeviceMatrixStorage device_b_storage{shape.k, shape.n};
    PitchedDeviceMatrixStorage device_c_storage{shape.m, shape.n};

    std::mt19937_64 generator{config.random_seed};
    auto host_a = make_matrix_view(host_a_storage.get(), shape.m, shape.k);
    auto host_b = make_matrix_view(host_b_storage.get(), shape.k, shape.n);
    auto host_reference_output = make_matrix_view(host_reference_storage.get(), shape.m, shape.n);
    fill_random_matrix(host_a, generator);
    fill_random_matrix(host_b, generator);
    compute_cpu_reference(host_a, host_b, host_reference_output);
    copy_matrix_async(host_a, device_a_storage.view(), cudaMemcpyHostToDevice, stream);
    copy_matrix_async(host_b, device_b_storage.view(), cudaMemcpyHostToDevice, stream);

    const auto host_reference = make_matrix_view<const double>(host_reference_storage.get(), shape.m, shape.n);
    const auto device_a = device_a_storage.const_view();
    const auto device_b = device_b_storage.const_view();
    auto device_c = device_c_storage.view();
    validate_benchmark_matrices(host_reference, device_c);

    std::vector<BenchmarkResult> results;
    results.reserve(cases.size());
    for (const auto& benchmark_case : cases) {
        auto matmul = impl_->callback_factory.make(benchmark_case.version, benchmark_case.topology);
        device_c_storage.zero_async(stream);
        results.push_back(
            run_case(benchmark_case, config, matmul, host_reference, device_a, device_b, device_c, stream));
    }

    return results;
}

void write_report(std::ostream& out, std::string_view stage_title, MatmulShape shape, const BenchmarkConfig& config,
                  std::span<const BenchmarkResult> results) {
    const bool has_topology = std::any_of(results.begin(), results.end(),
                                          [](const BenchmarkResult& result) { return result.topology.has_value(); });

    out << std::format("## {}\n\n", stage_title);
    out << std::format("M={}, N={}, K={}, warmup={}, timed={}, seed=0x{:X}\n\n", shape.m, shape.n, shape.k,
                       config.warmup_iterations, config.timed_iterations, config.random_seed);

    std::vector<std::string> headers{"Name",    "Mean (ms)",   "Median (ms)", "Min (ms)",
                                     "GFLOP/s", "Max Abs Err", "Max Rel Err", "Correct"};
    if (has_topology) {
        headers.insert(headers.end(), {"Block", "Tile"});
    }

    std::vector<std::vector<std::string>> rows;
    rows.reserve(results.size());
    for (const auto& result : results) {
        std::vector<std::string> row{
            result.name,
            std::format("{:.4f}", result.mean_ms),
            std::format("{:.4f}", result.median_ms),
            std::format("{:.4f}", result.min_ms),
            std::format("{:.2f}", result.gflops),
            std::format("{:.3e}", result.max_abs_error),
            std::format("{:.3e}", result.max_rel_error),
            result.correctness_passed ? "yes" : "no",
        };

        if (has_topology) {
            if (result.topology) {
                const auto& tile = result.topology->tile;
                row.push_back(std::format("{}x{}", result.topology->block.x, result.topology->block.y));
                row.push_back(tile ? std::format("{}x{}x{}", tile->m, tile->n, tile->k) : "-");
            } else {
                row.insert(row.end(), {"-", "-"});
            }
        }

        rows.push_back(std::move(row));
    }

    std::vector<std::size_t> column_widths(headers.size());
    for (std::size_t column = 0; column < headers.size(); ++column) {
        column_widths[column] = headers[column].size();
        for (const auto& row : rows) {
            column_widths[column] = std::max(column_widths[column], row[column].size());
        }
    }

    const auto write_row = [&](const std::vector<std::string>& cells) {
        out << '|';
        for (std::size_t column = 0; column < cells.size(); ++column) {
            out << ' ' << std::format("{:<{}}", cells[column], column_widths[column]) << " |";
        }
        out << '\n';
    };

    write_row(headers);

    out << '|';
    for (std::size_t width : column_widths) {
        out << ' ' << std::string(width, '-') << " |";
    }
    out << '\n';

    for (const auto& row : rows) {
        write_row(row);
    }
    out << '\n';
}

void write_csv_report(std::ostream& out, MatmulShape shape, std::span<const BenchmarkResult> results) {
    out << "Name,M,N,K,MeanMs,MedianMs,MinMs,GFLOPs,MaxAbsErr,MaxRelErr,Correct,BlockX,BlockY,TileM,TileN,TileK\n";

    for (const auto& result : results) {
        out << result.name << ',' << shape.m << ',' << shape.n << ',' << shape.k << ',';
        out << std::format("{:.6f},{:.6f},{:.6f},{:.6f}", result.mean_ms, result.median_ms, result.min_ms,
                           result.gflops);
        out << ',' << std::format("{:.6e},{:.6e}", result.max_abs_error, result.max_rel_error);
        out << ',' << (result.correctness_passed ? "true" : "false") << ',';

        if (result.topology) {
            out << result.topology->block.x;
        }
        out << ',';
        if (result.topology) {
            out << result.topology->block.y;
        }
        out << ',';
        if (result.topology && result.topology->tile) {
            out << result.topology->tile->m;
        }
        out << ',';
        if (result.topology && result.topology->tile) {
            out << result.topology->tile->n;
        }
        out << ',';
        if (result.topology && result.topology->tile) {
            out << result.topology->tile->k;
        }
        out << '\n';
    }
}

} // namespace cuda_matmul_lab
