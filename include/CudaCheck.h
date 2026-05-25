#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d in call '%s'\n      %s\n",          \
              __FILE__, __LINE__, #call, cudaGetErrorString(err));             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_LAST()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d\n          %s\n", __FILE__,         \
              __LINE__, cudaGetErrorString(err));                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
