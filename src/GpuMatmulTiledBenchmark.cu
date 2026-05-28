#include "CudaCheck.h"
#include "GpuMatmulTiledBenchmark.h"
#include <cstddef>
#include <sstream>

#define TILE_SIZE 16

__global__ void matmul_tiled(const float *a, const float *b, float *c, int n) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {

    // Load tile of A into As
    int aRow = blockIdx.y * TILE_SIZE + ty;
    int aCol = t * TILE_SIZE + tx;

    if (aRow < n && aCol < n) {
      As[ty][tx] = a[aRow * n + aCol];
    } else {
      As[ty][tx] = 0.0f;
    }

    // Load tile of B into Bs
    int bRow = t * TILE_SIZE + ty;
    int bCol = blockIdx.x * TILE_SIZE + tx;

    if (bRow < n && bCol < n) {
      Bs[ty][tx] = b[bRow * n + bCol];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < n && col < n)
    c[row * n + col] = sum;
}

GpuMatmulTiledBenchmark::GpuMatmulTiledBenchmark(std::size_t n) : n_{n} {};

std::string GpuMatmulTiledBenchmark::name() const { return "gpu-matmul-tiled"; }

void GpuMatmulTiledBenchmark::setup() {
  const std::size_t total = n_ * n_;

  a_.assign(total, 1.0f);
  b_.assign(total, 1.0f);
  c_.assign(total, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a_, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c_, total * sizeof(float)));
}

void GpuMatmulTiledBenchmark::run() {
  const std::size_t total = n_ * n_;

  CUDA_CHECK(cudaMemcpy(d_a_, a_.data(), total * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_, b_.data(), total * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((n_ + TILE_SIZE - 1) / TILE_SIZE, (n_ + TILE_SIZE - 1) / TILE_SIZE);

  matmul_tiled<<<grid, block>>>(d_a_, d_b_, d_c_, static_cast<int>(n_));

  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(c_.data(), d_c_, total * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void GpuMatmulTiledBenchmark::verify() {
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

void GpuMatmulTiledBenchmark::teardown() {
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

double GpuMatmulTiledBenchmark::flops_per_iteration() const {
  const double n = static_cast<double>(n_);
  return 2.0 * n * n * n;
}
