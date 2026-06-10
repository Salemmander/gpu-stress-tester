#include "CudaCheck.h"
#include "GpuMemcpyBandwidthBenchmark.h"
#include <cstddef>
#include <stdexcept>

__global__ void copy_kernel(const float *src, float *dst, std::size_t n) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i];
  }
}

GpuMemcpyBandwidthBenchmark::GpuMemcpyBandwidthBenchmark(std::size_t n)
    : n_(n) {};

std::string GpuMemcpyBandwidthBenchmark::name() const {
  return "gpu-memcpy-bandwidth";
}

void GpuMemcpyBandwidthBenchmark::setup() {
  src_.assign(n_, 1.0f);
  dst_.assign(n_, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_src_, n_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst_, n_ * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_src_, src_.data(), n_ * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void GpuMemcpyBandwidthBenchmark::run() {
  dim3 block(256);
  dim3 grid((n_ + block.x - 1) / block.x);

  copy_kernel<<<grid, block>>>(d_src_, d_dst_, n_);

  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
}
void GpuMemcpyBandwidthBenchmark::verify() {

  CUDA_CHECK(cudaMemcpy(dst_.data(), d_dst_, n_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
  if (dst_ != src_) {
    throw std::runtime_error("Verification failed");
  }
}
void GpuMemcpyBandwidthBenchmark::teardown() {
  src_.clear();
  dst_.clear();

  if (d_src_)
    CUDA_CHECK(cudaFree(d_src_));
  if (d_dst_)
    CUDA_CHECK(cudaFree(d_dst_));

  d_src_ = nullptr;
  d_dst_ = nullptr;
}

double GpuMemcpyBandwidthBenchmark::bytes_per_iteration() const {
  // kernel reads src: n floats
  // kernel writes dst: n floats
  // therefore the 2.0
  return 2.0 * static_cast<double>(n_) * sizeof(float);
}
