#include <cstdlib>
#include <exception>
#include <format>
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

enum class ReportFormat {
    MARKDOWN,
    CSV,
};

constexpr std::string_view usage =
    "Usage: benchmark_runner [options]\n"
    "  --m <n>            Rows of A and C (default 1024)\n"
    "  --n <n>            Columns of B and C (default 1024)\n"
    "  --k <n>            Columns of A / rows of B (default 1024)\n"
    "  --warmup <n>       Warmup iterations per case (default 5)\n"
    "  --iterations <n>   Timed iterations per case (default 5)\n"
    "  --seed <n>         Random seed for input generation (default 0xC0FFEE)\n"
    "  --abs-tol <f>      Absolute error tolerance (default 1e-4)\n"
    "  --rel-tol <f>      Relative error tolerance (default 1e-3)\n"
    "  --device <n>       CUDA device index to run on (default: active device)\n"
    "  --title <s>        Report title (default \"cuda-matmul-lab benchmark\"); ignored for --format csv\n"
    "  --format <fmt>     Report format: markdown or csv (default markdown)\n"
    "  --cases <path>     Load benchmark cases from a CSV file instead of the built-in list. Header row:\n"
    "                     Version,BlockX,BlockY,TileM,TileN,TileK\n"
    "                     Version is a matmul name (e.g. \"naive\", \"cublas\"). All other columns are unsigned\n"
    "                     integers; leave BlockX/BlockY blank for a version with no topology (e.g. cublas), and\n"
    "                     leave TileM/TileN/TileK blank together to omit that optional field.\n"
    "  --output <path>    Write the report to a file instead of stdout\n"
    "  --help             Show this message\n";

struct CliOptions {
    MatmulShape shape;
    BenchmarkConfig config;
    std::optional<int> device;
    std::string title = "cuda-matmul-lab benchmark";
    std::optional<std::string> output_path;
    std::optional<std::string> cases_path;
    ReportFormat format = ReportFormat::MARKDOWN;
};

[[nodiscard]] std::string next_value(int argc, char** argv, int& i, std::string_view flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument{std::format("missing value for {}", flag)};
    }
    return argv[++i];
}

[[nodiscard]] std::string trim(std::string_view text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return std::string{text.substr(start, end - start + 1)};
}

// Splits on commas without any quoting support, preserving trailing empty fields (e.g. "a,b," -> {"a", "b", ""}).
[[nodiscard]] std::vector<std::string> split_csv_line(std::string_view line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        if (comma == std::string_view::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, comma - start));
        start = comma + 1;
    }
    return fields;
}

[[nodiscard]] std::optional<unsigned> parse_optional_unsigned(const std::string& field) {
    if (field.empty()) {
        return std::nullopt;
    }
    return static_cast<unsigned>(std::stoul(field));
}

// Reverse lookup for the CSV's Version column: matches against the same names get_matmul_name produces, so there is
// only ever one table of matmul version names in this codebase.
[[nodiscard]] MatmulVersion parse_matmul_version(std::string_view name) {
    for (int i = 0; i < static_cast<int>(MatmulVersion::COUNT); ++i) {
        const auto version = static_cast<MatmulVersion>(i);
        if (cuda_matmul_lab::get_matmul_name(version) == name) {
            return version;
        }
    }
    throw std::invalid_argument{std::format("unknown matmul version name: {}", name)};
}

[[nodiscard]] BenchmarkCase parse_benchmark_case_row(const std::vector<std::string>& fields, std::size_t row_number) {
    if (fields.size() != 6) {
        throw std::invalid_argument{
            std::format("cases CSV row {}: expected 6 columns, got {}", row_number, fields.size())};
    }

    const MatmulVersion version = parse_matmul_version(trim(fields[0]));
    const auto block_x = parse_optional_unsigned(trim(fields[1]));
    const auto block_y = parse_optional_unsigned(trim(fields[2]));
    const auto tile_m = parse_optional_unsigned(trim(fields[3]));
    const auto tile_n = parse_optional_unsigned(trim(fields[4]));
    const auto tile_k = parse_optional_unsigned(trim(fields[5]));

    if (!block_x && !block_y) {
        return BenchmarkCase{.version = version, .topology = std::nullopt};
    }
    if (!block_x || !block_y) {
        throw std::invalid_argument{
            std::format("cases CSV row {}: BlockX and BlockY must both be set or both blank", row_number)};
    }

    std::optional<TileShape> tile;
    if (tile_m || tile_n || tile_k) {
        if (!tile_m || !tile_n || !tile_k) {
            throw std::invalid_argument{
                std::format("cases CSV row {}: TileM, TileN, and TileK must all be set or all blank", row_number)};
        }
        tile = TileShape{.m = *tile_m, .n = *tile_n, .k = *tile_k};
    }

    return BenchmarkCase{
        .version = version,
        .topology = LaunchTopology{.block = BlockShape{.x = *block_x, .y = *block_y}, .tile = tile},
    };
}

[[nodiscard]] std::vector<BenchmarkCase> read_benchmark_cases_csv(const std::string& path) {
    std::ifstream file{path};
    if (!file) {
        throw std::runtime_error{std::format("failed to open cases file: {}", path)};
    }

    std::vector<BenchmarkCase> cases;
    std::string line;
    std::size_t row_number = 0;
    bool header_skipped = false;
    while (std::getline(file, line)) {
        ++row_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (!header_skipped) {
            header_skipped = true;
            continue;
        }
        cases.push_back(parse_benchmark_case_row(split_csv_line(line), row_number));
    }
    return cases;
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
        } else if (arg == "--cases") {
            options.cases_path = next_value(argc, argv, i, arg);
        } else if (arg == "--format") {
            const std::string value = next_value(argc, argv, i, arg);
            if (value == "markdown") {
                options.format = ReportFormat::MARKDOWN;
            } else if (value == "csv") {
                options.format = ReportFormat::CSV;
            } else {
                throw std::invalid_argument{
                    std::format("unrecognized --format value (expected markdown or csv): {}", value)};
            }
        } else {
            throw std::invalid_argument{std::format("unrecognized option: {}", arg)};
        }
    }

    return options;
}

// One entry per implemented matmul version, with the topology tuned for that kernel. Add a case here when a new
// kernel stage from the README roadmap is implemented. Overridden entirely by --cases when given.
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
            .topology =
                LaunchTopology{.block = BlockShape{.x = 32, .y = 8}, .tile = TileShape{.m = 8, .n = 32, .k = 8}},
        },
        BenchmarkCase{
            .version = MatmulVersion::TILED,
            .topology =
                LaunchTopology{.block = BlockShape{.x = 32, .y = 16}, .tile = TileShape{.m = 16, .n = 32, .k = 16}},
        },
        BenchmarkCase{
            .version = MatmulVersion::TILED,
            .topology =
                LaunchTopology{.block = BlockShape{.x = 32, .y = 32}, .tile = TileShape{.m = 32, .n = 32, .k = 32}},
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
        throw std::runtime_error{std::format("cudaSetDevice failed: {}", cudaGetErrorString(status))};
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
        const auto cases =
            options->cases_path ? read_benchmark_cases_csv(*options->cases_path) : make_benchmark_cases();
        const auto results = session.run(options->shape, cases, options->config, cudaStream_t{nullptr});

        const auto write = [&](std::ostream& out) {
            if (options->format == ReportFormat::CSV) {
                cuda_matmul_lab::write_csv_report(out, options->shape, results);
            } else {
                cuda_matmul_lab::write_report(out, options->title, options->shape, options->config, results);
            }
        };

        if (options->output_path) {
            std::ofstream file{*options->output_path};
            if (!file) {
                throw std::runtime_error{std::format("failed to open output file: {}", *options->output_path)};
            }
            write(file);
        } else {
            write(std::cout);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n" << usage;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
