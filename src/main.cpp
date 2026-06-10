#include "BenchmarkRunner.h"
#include "CpuMatmulBenchmark.h"
#include "GpuMatmulBenchmark.h"
#include "GpuMatmulTiledBenchmark.h"
#include "GpuMemcpyBandwidthBenchmark.h"
#include "GpuVectorAddBenchmark.h"

#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
  std::string benchmark_name;
  std::size_t size = 0;
  std::size_t iterations = 0;

  for (int i = 1; i < argc; i += 2) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) {
      std::cerr << "Missing Value for " << flag << "\n";
      return 1;
    }

    const std::string value = argv[i + 1];

    if (flag == "--benchmark") {
      benchmark_name = value;
    } else if (flag == "--size") {
      size = std::stoull(value);
    } else if (flag == "--iterations") {
      iterations = std::stoull(value);
    } else {
      std::cerr << "Unknown flag: " << flag << "\n";
      return 1;
    }
  }

  if (benchmark_name.empty() || size == 0 || iterations == 0) {
    std::cerr << "Usage: gpu-stress --benchmark <name> "
              << "--size <N> --iterations <N>\n";
    return 1;
  }

  // clang-format off
  std::map<std::string, std::function<std::unique_ptr<Benchmark>(std::size_t)>> benchmarks = {
      {"cpu-matmul",       [](std::size_t s){ return std::make_unique<CpuMatmulBenchmark>(s); }},
      {"gpu-vector-add",   [](std::size_t s){ return std::make_unique<GpuVectorAddBenchmark>(s); }},
      {"gpu-matmul",       [](std::size_t s){ return std::make_unique<GpuMatmulBenchmark>(s); }},
      {"gpu-matmul-tiled", [](std::size_t s){ return std::make_unique<GpuMatmulTiledBenchmark>(s); }},
      {"gpu-memcpy-bandwidth", [](std::size_t s){ return std::make_unique<GpuMemcpyBandwidthBenchmark>(s); }},
  };
  // clang-format on

  std::unique_ptr<Benchmark> benchmark;
  auto it = benchmarks.find(benchmark_name);
  if (it != benchmarks.end()) {
    benchmark = it->second(size);
  } else {
    std::cerr << "Unknown Benchmark: " << benchmark_name << "\n";
    return 1;
  }

  BenchmarkRunner runner{iterations};

  try {
    const auto result = runner.run(*benchmark);

    std::cout << "Benchmark: " << benchmark->name() << "\n"
              << "Size: " << size << "\n"
              << "Iterations: " << result.iterations << "\n"
              << "Mean: " << result.mean_seconds << " s\n"
              << "Stddev: " << result.stddev_seconds << " s\n";

    if (result.mean_tflops > 0.0)
      std::cout << "TFLOPS: " << result.mean_tflops << "\n";

    if (result.mean_gbps > 0.0)
      std::cout << "GB/s: " << result.mean_gbps << "\n";

  } catch (const std::exception &e) {
    std::cerr << "Benchmark failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
