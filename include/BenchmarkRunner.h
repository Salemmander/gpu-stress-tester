#pragma once

#include "Benchmark.h"
#include <cstddef>

class BenchmarkRunner {
public:
  struct Result {
    std::size_t iterations;
    double mean_seconds;
    double stddev_seconds;
    double mean_tflops;
    double mean_gbps;
  };

  explicit BenchmarkRunner(std::size_t iterations);
  Result run(Benchmark &benchmark);

private:
  std::size_t iterations_;
};
