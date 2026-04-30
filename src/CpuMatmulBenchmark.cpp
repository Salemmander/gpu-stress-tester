#include "CpuMatmulBenchmark.h"
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

CpuMatmulBenchmark::CpuMatmulBenchmark(std::size_t n) : n_{n} {};

std::string CpuMatmulBenchmark::name() const { return "cpu-matmul"; };

void CpuMatmulBenchmark::setup() {
  a_.assign(n_ * n_, 1.0f);
  b_.assign(n_ * n_, 1.0f);
  c_.assign(n_ * n_, 0.0f);
}

void CpuMatmulBenchmark::run() {
  for (std::size_t i = 0; i < n_; ++i) {
    for (std::size_t j = 0; j < n_; ++j) {
      float sum = 0.0f;
      for (std::size_t k = 0; k < n_; ++k)
        sum += a_[i * n_ + k] * b_[k * n_ + j];
      c_[i * n_ + j] = sum;
    }
  }
}

void CpuMatmulBenchmark::verify() {
  const float expected = static_cast<float>(n_);
  const float tolerance = expected * 1e-3f;

  for (std::size_t i = 0; i < n_ * n_; ++i) {
    if (std::abs(c_[i] - expected) > tolerance) {
      std::ostringstream oss;
      oss << "Verification failed at index " << i << ": got " << c_[i]
          << ", expected " << expected;
      throw std::runtime_error(oss.str());
    }
  }
}

void CpuMatmulBenchmark::teardown() {
  a_.clear();
  b_.clear();
  c_.clear();
}
