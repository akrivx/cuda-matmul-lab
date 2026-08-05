#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark.hpp"
#include "launch_topology.hpp"
#include "matmul.hpp"

namespace {

using cuda_matmul_lab::BenchmarkCase;
using cuda_matmul_lab::BenchmarkConfig;
using cuda_matmul_lab::BenchmarkSession;
using cuda_matmul_lab::BlockShape;
using cuda_matmul_lab::LaunchTopology;
using cuda_matmul_lab::MatmulShape;
using cuda_matmul_lab::MatmulVersion;
using cuda_matmul_lab::TileShape;

constexpr std::string_view usage = "Usage: benchmark_runner [options]\n"
                                   "  --m <n>            Rows of A and C (default 1024)\n"
                                   "  --n <n>            Columns of B and C (default 1024)\n"
                                   "  --k <n>            Columns of A / rows of B (default 1024)\n"
                                   "  --warmup <n>       Warmup iterations per case (default 5)\n"
                                   "  --iterations <n>   Timed iterations per case (default 5)\n"
                                   "  --seed <n>         Random seed for input generation (default 0xC0FFEE)\n"
                                   "  --abs-tol <f>      Absolute error tolerance (default 1e-4)\n"
                                   "  --rel-tol <f>      Relative error tolerance (default 1e-3)\n"
                                   "  --device <n>       CUDA device index to run on (default: active device)\n"
                                   "  --title <s>        Report title (default \"cuda-matmul-lab benchmark\")\n"
                                   "  --output <path>    Write the report to a file instead of stdout\n"
                                   "  --help             Show this message\n";

struct CliOptions {
    MatmulShape shape;
    BenchmarkConfig config;
    std::optional<int> device;
    std::string title = "cuda-matmul-lab benchmark";
    std::optional<std::string> output_path;
};

[[nodiscard]] std::string next_value(int argc, char** argv, int& i, std::string_view flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument{std::string{"missing value for "} + std::string{flag}};
    }
    return argv[++i];
}

// Returns std::nullopt when the user only asked for --help.
[[nodiscard]] std::optional<CliOptions> parse_args(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << usage;
            return std::nullopt;
        }
        if (arg == "--m") {
            options.shape.m = std::stoull(next_value(argc, argv, i, arg));
        } else if (arg == "--n") {
            options.shape.n = std::stoull(next_value(argc, argv, i, arg));
        } else if (arg == "--k") {
            options.shape.k = std::stoull(next_value(argc, argv, i, arg));
        } else if (arg == "--warmup") {
            options.config.warmup_iterations = std::stoull(next_value(argc, argv, i, arg));
        } else if (arg == "--iterations") {
            options.config.timed_iterations = std::stoull(next_value(argc, argv, i, arg));
        } else if (arg == "--seed") {
            options.config.random_seed = std::stoull(next_value(argc, argv, i, arg), nullptr, 0);
        } else if (arg == "--abs-tol") {
            options.config.absolute_tolerance = std::stod(next_value(argc, argv, i, arg));
        } else if (arg == "--rel-tol") {
            options.config.relative_tolerance = std::stod(next_value(argc, argv, i, arg));
        } else if (arg == "--device") {
            options.device = std::stoi(next_value(argc, argv, i, arg));
        } else if (arg == "--title") {
            options.title = next_value(argc, argv, i, arg);
        } else if (arg == "--output") {
            options.output_path = next_value(argc, argv, i, arg);
        } else {
            throw std::invalid_argument{std::string{"unrecognized option: "} + std::string{arg}};
        }
    }

    return options;
}

// One entry per implemented matmul version, with the topology tuned for that kernel. Add a case here when a new
// kernel stage from the README roadmap is implemented.
[[nodiscard]] std::vector<BenchmarkCase> make_benchmark_cases() {
    return {
        BenchmarkCase{
            .version = MatmulVersion::NAIVE,
            .topology = LaunchTopology{.block = BlockShape{.x = 32, .y = 8}},
        },
        BenchmarkCase{
            .version = MatmulVersion::NAIVE_COALESCED,
            .topology = LaunchTopology{.block = BlockShape{.x = 32, .y = 8}},
        },
        BenchmarkCase{
            .version = MatmulVersion::TILED,
            .topology = LaunchTopology{.block = BlockShape{.x = 32, .y = 8}},
        },
        BenchmarkCase{
            .version = MatmulVersion::TILED,
            .topology = LaunchTopology{.block = BlockShape{.x = 32, .y = 16}},
        },
        BenchmarkCase{
            .version = MatmulVersion::TILED,
            .topology = LaunchTopology{.block = BlockShape{.x = 32, .y = 32}},
        },
        BenchmarkCase{
            .version = MatmulVersion::THREAD_TILED,
            .topology =
                LaunchTopology{.block = BlockShape{.x = 16, .y = 8}, .tile = TileShape{.m = 16, .n = 8, .k = 8}},
        },
        BenchmarkCase{
            .version = MatmulVersion::CUBLAS,
            .topology = std::nullopt,
        },
    };
}

void select_device(int device) {
    const cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        throw std::runtime_error{std::string{"cudaSetDevice failed: "} + cudaGetErrorString(status)};
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        if (!options) {
            return EXIT_SUCCESS;
        }

        if (options->device) {
            select_device(*options->device);
        }

        BenchmarkSession session;
        const auto cases = make_benchmark_cases();
        const auto results = session.run(options->shape, cases, options->config, cudaStream_t{nullptr});

        if (options->output_path) {
            std::ofstream file{*options->output_path};
            if (!file) {
                throw std::runtime_error{"failed to open output file: " + *options->output_path};
            }
            cuda_matmul_lab::write_report(file, options->title, options->shape, options->config, results);
        } else {
            cuda_matmul_lab::write_report(std::cout, options->title, options->shape, options->config, results);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n" << usage;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
