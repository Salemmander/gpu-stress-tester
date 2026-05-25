#pragma once

#include "Benchmark.h"

#include <cstddef>
#include <string>
#include <vector>

class GpuVectorAddBenchmark : public Benchmark {
public:
  explicit GpuVectorAddBenchmark(std::size_t n);

  std::string name() const override;
  void setup() override;
  void run() override;
  void verify() override;
  void teardown() override;

private:
  std::size_t n_;
  std::vector<float> a_;
  std::vector<float> b_;
  std::vector<float> c_;

  float* d_a_ = nullptr;
  float* d_b_ = nullptr;
  float* d_c_ = nullptr;

  friend class GpuVectorAddTestHelper;
};
