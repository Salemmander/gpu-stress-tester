#include "CudaCheck.h"
#include "GpuVectorAddBenchmark.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

GpuVectorAddBenchmark::GpuVectorAddBenchmark(std::size_t n) : n_{n} {};

std::string GpuVectorAddBenchmark::name() const { return "gpu-vector-add"; };

void GpuVectorAddBenchmark::setup() {
  a_.assign(n_, 1.0f);
  b_.assign(n_, 1.0f);
  c_.assign(n_, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a_, n_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_, n_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c_, n_ * sizeof(float)));
}

void GpuVectorAddBenchmark::run() {
  CUDA_CHECK(
      cudaMemcpy(d_a_, a_.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b_, b_.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (n_ + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a_, d_b_, d_c_, n_);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(c_.data(), d_c_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void GpuVectorAddBenchmark::verify() {
  const float expected = 2.0f;
  const float tolerance = expected * 1e-3f;

  for (std::size_t i = 0; i < n_; ++i) {
    if (std::abs(c_[i] - expected) > tolerance) {
      std::ostringstream oss;
      oss << "Verification failed at index " << i << ": got " << c_[i]
          << ", expected " << expected;
      throw std::runtime_error(oss.str());
    }
  }
}

void GpuVectorAddBenchmark::teardown() {
  a_.clear();
  b_.clear();
  c_.clear();

  if (d_a_)
    CUDA_CHECK(cudaFree(d_a_));
  if (d_b_)
    CUDA_CHECK(cudaFree(d_b_));
  if (d_c_)
    CUDA_CHECK(cudaFree(d_c_));

  d_a_ = nullptr;
  d_b_ = nullptr;
  d_c_ = nullptr;
}
