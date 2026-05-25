#include "CudaCheck.h"
#include "GpuMatmulBenchmark.h"
#include <cmath>
#include <cstddef>
#include <sstream>

__global__ void matmul(const float *a, const float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= n || col >= n)
    return;

  float sum = 0.0f;
  for (int k = 0; k < n; ++k) {
    sum += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = sum;
}

GpuMatmulBenchmark::GpuMatmulBenchmark(std::size_t n) : n_{n} {};

std::string GpuMatmulBenchmark::name() const { return "gpu-matmul"; }

void GpuMatmulBenchmark::setup() {
  const std::size_t total = n_ * n_;

  a_.assign(total, 1.0f);
  b_.assign(total, 1.0f);
  c_.assign(total, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a_, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c_, total * sizeof(float)));
}

void GpuMatmulBenchmark::run() {
  const std::size_t total = n_ * n_;

  CUDA_CHECK(cudaMemcpy(d_a_, a_.data(), total * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_, b_.data(), total * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((n_ + block.x - 1) / block.x, (n_ + block.y - 1) / block.y);

  matmul<<<grid, block>>>(d_a_, d_b_, d_c_, static_cast<int>(n_));

  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(c_.data(), d_c_, total * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void GpuMatmulBenchmark::verify() {
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

void GpuMatmulBenchmark::teardown() {
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

double GpuMatmulBenchmark::flops_per_iteration() const {
  const double n = static_cast<double>(n_);
  return 2.0 * n * n * n;
}
