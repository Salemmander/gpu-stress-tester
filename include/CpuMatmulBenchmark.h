#pragma once

#include "Benchmark.h"

#include <cstddef>
#include <string>
#include <vector>

class CpuMatmulBenchmark : public Benchmark {
public:
  explicit CpuMatmulBenchmark(std::size_t n);

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

  friend class CpuMatmulTestHelper;
};
