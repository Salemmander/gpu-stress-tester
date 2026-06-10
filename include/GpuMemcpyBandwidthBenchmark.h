#pragma once

#include "Benchmark.h"

#include <cstddef>
#include <string>
#include <vector>

class GpuMemcpyBandwidthBenchmark : public Benchmark {
public:
  explicit GpuMemcpyBandwidthBenchmark(std::size_t n);

  std::string name() const override;
  void setup() override;
  void run() override;
  void verify() override;
  void teardown() override;
  double bytes_per_iteration() const override;

private:
  std::size_t n_;
  std::vector<float> src_;
  std::vector<float> dst_;

  float *d_src_ = nullptr;
  float *d_dst_ = nullptr;

  friend class GpuMemcpyBandwidthTestHelper;
};
