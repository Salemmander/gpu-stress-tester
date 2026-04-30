#include "BenchmarkRunner.h"
#include "CpuMatmulBenchmark.h"

#include <cstddef>
#include <exception>
#include <iostream>
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

  std::unique_ptr<Benchmark> benchmark;
  if (benchmark_name == "cpu-matmul") {
    benchmark = std::make_unique<CpuMatmulBenchmark>(size);
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
  } catch (const std::exception &e) {
    std::cerr << "Benchmark failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
