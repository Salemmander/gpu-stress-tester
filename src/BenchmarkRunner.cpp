#include "BenchmarkRunner.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

BenchmarkRunner::BenchmarkRunner(std::size_t iterations)
    : iterations_{iterations} {}

BenchmarkRunner::Result BenchmarkRunner::run(Benchmark &benchmark) {
  // === 1. Collect timing samples ===
  std::vector<double> samples;
  samples.reserve(iterations_);

  for (std::size_t i = 0; i < iterations_; ++i) {
    benchmark.setup();

    const auto start = std::chrono::steady_clock::now();
    benchmark.run();
    const auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = end - start;
    samples.push_back(elapsed.count());

    benchmark.verify();
    benchmark.teardown();
  }

  // === 2. Compute mean time ===
  double sum = 0.0;
  for (double s : samples)
    sum += s;
  const double mean = sum / static_cast<double>(iterations_);

  // === 3. Compute standard deviation ===
  double sq_diff_sum = 0.0;
  for (double s : samples) {
    const double diff = s - mean;
    sq_diff_sum += diff * diff;
  }

  const double stddev =
      (iterations_ > 1)
          ? std::sqrt(sq_diff_sum / static_cast<double>(iterations_ - 1))
          : 0.0;

  // === 4. Compute TFLOPS (only if the benchmark reports FLOPs) ===
  double mean_tflops = 0.0;
  const double flops_per_iter = benchmark.flops_per_iteration();
  if (flops_per_iter > 0.0 && mean > 0.0) {
    mean_tflops = (flops_per_iter / mean) / 1e12;
  }

  double mean_gbps = 0.0;
  const double bytes_per_iter = benchmark.bytes_per_iteration();
  if (bytes_per_iter > 0.0 && mean > 0.0)
    mean_gbps = (bytes_per_iter / mean) / 1e9;

  return Result{iterations_, mean, stddev, mean_tflops, mean_gbps};
}
