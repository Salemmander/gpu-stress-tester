# gpu-stress-tester — Implementation Plan

This document contains the detailed phased roadmap for the project. It is the canonical reference for what will be built and the order in which features will be delivered.

Each phase is roughly one to two evenings of focused work. Earlier phases are pure modern C++; CUDA enters in Phase 1 and grows from there.

## Project Approach

The primary objective of this project is to build a stress and validation framework capable of pushing GPUs hard and verifying they remain functional and correct under sustained heavy load — the kind of workloads a GPU silicon validation or systems software engineer would use to ensure reliability in real deployments.

As a supporting activity, selected phases include deliberate kernel optimization work. The goal of this optimization is not only to improve performance, but to develop the skills and techniques required to write *efficiently stressful* workloads (for example, kernels that maximize memory throughput, sustain high utilization, or apply pathological access patterns).

This dual-track approach allows meaningful CUDA learning while keeping the overall project aligned with diagnostic, validation, and stress-engineering work.

## Phase 0 - C++ benchmarking framework (CPU only)

Build the skeleton of the whole tool with no GPU code at all.

- `Benchmark` abstract base class with `setup()`, `run()`, `verify()`, `teardown()`, and `name()` virtual methods
- Concrete `CpuMatmulBenchmark` (naive triple loop)
- `BenchmarkRunner` that times runs with `std::chrono`, computes mean and stddev over N iterations
- CLI with hand-rolled argument parsing (`--benchmark`, `--size`, `--iterations`)
- CMake build system, GoogleTest for unit tests
- Proper use of modern C++ (RAII, exceptions, smart pointers, etc.)

C++ concepts exercised: virtual functions, RAII, `std::chrono`, exceptions, `std::vector`, modern initialization, testing with GoogleTest.

## Phase 1 - Hello CUDA

- Install CUDA Toolkit, verify `nvcc` works against the RTX 3080
- Write `vector_add.cu`: add two arrays on the GPU, copy back, verify
- Wire it into the existing framework as a `GpuVectorAddBenchmark` subclass
- Add a CUDA error-checking macro and use it everywhere

CUDA concepts: kernel launch syntax, host/device memory, `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, error handling.

## Phase 2 - Naive matrix multiply

- Implement `GpuMatmulNaiveBenchmark`: each thread computes one output element
- Verify correctness against the CPU version from Phase 0
- Measure GFLOPS. It will be far below peak. That is the point.

This phase also serves as a vehicle to learn core CUDA programming patterns (thread indexing, memory access, basic kernel structure) that will later be applied when constructing more aggressive and stressful GPU workloads.

## Phase 3 - Tiled matrix multiply with shared memory

- Implement `GpuMatmulTiledBenchmark` using shared memory tiling
- Compare GFLOPS to Phase 2 (expected 5-10x speedup)
- Profile with Nsight Compute to confirm the gain comes from reduced global loads

CUDA concepts: shared memory, `__syncthreads`, tile-based decomposition, occupancy, memory hierarchy.

The optimization techniques learned here (coalesced access, data reuse, memory hierarchy awareness) are directly applicable to writing kernels that can sustain very high memory or compute throughput during stress testing.

## Phase 4 - Memory bandwidth benchmark

- `GpuMemcpyBandwidthBenchmark`: simple device-to-device copy kernel
- Measure achieved GB/s, compare to the 3080's ~760 GB/s theoretical peak
- Try strided vs coalesced access patterns; watch the bandwidth collapse on strided

CUDA concepts: coalesced memory access, memory transactions, vector loads.

This phase begins the core stress work: deliberately pressuring the memory subsystem and understanding how access patterns affect sustained throughput.

## Phase 5 - PCIe bandwidth benchmark

- `PcieBandwidthBenchmark`: time `cudaMemcpy` host-to-device and device-to-host
- Compare pageable vs pinned memory (`cudaMallocHost`)
- Compare to PCIe Gen4 x16 theoretical (~32 GB/s)

CUDA concepts: pinned memory, async copies, streams.

## Phase 6 - Real CLI tool with NVML telemetry

- Polished CLI: select benchmarks, set sizes, set iteration counts, choose output format
- NVML integration: query GPU temperature, power draw, clock speeds, memory clock
- Sample telemetry during runs and include in the report
- JSON or pretty-printed text output

C++ concepts: integrating a C library cleanly, RAII wrappers around `nvml*` handles.

## Phase 7 - Stress mode

- Run benchmarks in a loop for N minutes on a background thread
- After each iteration, hash or checksum the result and compare to the first iteration
- Any divergence is logged as a potential correctness fault (this is what GPU diag teams care about)
- Track thermal throttling: log when clock speeds drop under load

C++ concepts: `std::thread`, condition variables, atomics, clean shutdown signaling.

This is the flagship validation phase: sustained heavy load combined with continuous correctness and health monitoring.

## References

- NVIDIA, "An Even Easier Introduction to CUDA"
- Kirk and Hwu, _Programming Massively Parallel Processors_
- NVIDIA CUDA C++ Best Practices Guide
- NVIDIA Nsight Compute documentation
- NVML developer reference
