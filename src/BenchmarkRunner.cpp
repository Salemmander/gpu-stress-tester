#include "BenchmarkRunner.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

BenchmarkRunner::BenchmarkRunner(std::size_t iterations)
    : iterations_{iterations} {}

BenchmarkRunner::Result BenchmarkRunner::run(Benchmark &benchmark) {
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

  double sum = 0.0;
  for (double s : samples)
    sum += s;
  const double mean = sum / static_cast<double>(iterations_);

  double sq_diff_sum = 0.0;
  for (double s : samples) {
    const double diff = s - mean;
    sq_diff_sum += diff * diff;
  }

  const double stddev =
      (iterations_ > 1)
          ? std::sqrt(sq_diff_sum / static_cast<double>(iterations_ - 1))
          : 0.0;

  return Result{iterations_, mean, stddev};
}
