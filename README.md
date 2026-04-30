# gpu-stress-tester

A CUDA-based GPU stress tester and microbenchmark suite written in modern C++.
Inspired by the kind of diagnostic tooling that GPU vendors use to validate silicon:
exercise compute units, memory subsystems, and PCIe transfer paths under load,
then report measured performance against theoretical peaks.

Target hardware for development: NVIDIA RTX 3080 (Ampere, sm_86, 10GB GDDR6X, PCIe Gen4 x16).

## Goals

- Build a real C++ project with proper structure (CMake, headers, tests, CLI).
- Learn CUDA hands-on by writing kernels that probe specific GPU subsystems.
- Measure and report: GFLOPS, GB/s, PCIe bandwidth, thermal/power telemetry.
- Detect correctness drift under sustained stress (a real GPU diagnostic concern).

## Why this project

The role this targets (NVIDIA System Software Engineer - GPU) builds the
"most stressful set of applications a GPU or HPC server would see in its life cycle."
This project is a small, honest version of that brief: write tools that push the GPU
hard and report what comes back.

## Phased plan

Each phase is roughly one to two evenings of focused work. Earlier phases are pure
modern C++; CUDA enters in Phase 1 and grows from there.

### Phase 0 - C++ benchmarking framework (CPU only)

Build the skeleton of the whole tool with no GPU code at all.

- `Benchmark` abstract base class with `run()`, `name()`, `report()` virtual methods
- Concrete `CpuMatmulBenchmark` (naive triple loop) and `CpuMemcpyBenchmark`
- `BenchmarkRunner` that times runs with `std::chrono`, computes mean and stddev over N iterations
- CLI with argument parsing (`cxxopts` or hand-rolled)
- CMake build system, GoogleTest for unit tests
- Logging, error handling with exceptions, RAII for resources

C++ concepts exercised: virtual functions, smart pointers, templates,
`std::chrono`, exceptions, `std::vector`, modern initialization.

### Phase 1 - Hello CUDA

- Install CUDA Toolkit, verify `nvcc` works against the 3080
- Write `vector_add.cu`: add two arrays on the GPU, copy back, verify
- Wire it into the existing framework as a `GpuVectorAddBenchmark` subclass
- Add a CUDA error-checking macro and use it everywhere

CUDA concepts: kernel launch syntax, host/device memory, `cudaMalloc`,
`cudaMemcpy`, `cudaDeviceSynchronize`, error handling.

### Phase 2 - Naive matrix multiply

- Implement `GpuMatmulNaiveBenchmark`: each thread computes one output element
- Verify correctness against the CPU version from Phase 0
- Measure GFLOPS. It will be far below peak. That is the point.

### Phase 3 - Tiled matrix multiply with shared memory

- Implement `GpuMatmulTiledBenchmark` using shared memory tiling
- Compare GFLOPS to Phase 2 (expected 5-10x speedup)
- Profile with Nsight Compute to confirm the gain comes from reduced global loads

CUDA concepts: shared memory, `__syncthreads`, tile-based decomposition,
occupancy, memory hierarchy.

### Phase 4 - Memory bandwidth benchmark

- `GpuMemcpyBandwidthBenchmark`: simple device-to-device copy kernel
- Measure achieved GB/s, compare to the 3080's ~760 GB/s theoretical peak
- Try strided vs coalesced access patterns; watch the bandwidth collapse on strided

CUDA concepts: coalesced memory access, memory transactions, vector loads.

### Phase 5 - PCIe bandwidth benchmark

- `PcieBandwidthBenchmark`: time `cudaMemcpy` host-to-device and device-to-host
- Compare pageable vs pinned memory (`cudaMallocHost`)
- Compare to PCIe Gen4 x16 theoretical (~32 GB/s)

CUDA concepts: pinned memory, async copies, streams.

### Phase 6 - Real CLI tool with NVML telemetry

- Polished CLI: select benchmarks, set sizes, set iteration counts, choose output format
- NVML integration: query GPU temperature, power draw, clock speeds, memory clock
- Sample telemetry during runs and include in the report
- JSON or pretty-printed text output

C++ concepts: integrating a C library cleanly, RAII wrappers around `nvml*` handles.

### Phase 7 - Stress mode

- Run benchmarks in a loop for N minutes on a background thread
- After each iteration, hash or checksum the result and compare to the first iteration
- Any divergence is logged as a potential correctness fault (this is what GPU diag teams care about)
- Track thermal throttling: log when clock speeds drop under load

C++ concepts: `std::thread`, condition variables, atomics, clean shutdown signaling.

## Build

```bash
mkdir build && cd build
cmake ..
make -j
./gpu-stress --help
```

(Phase 0 will not require CUDA. Later phases will gate the GPU benchmarks behind
`find_package(CUDAToolkit)` so the project still builds without CUDA installed.)

## Progress

- [ ] Phase 0 - C++ benchmarking framework
- [ ] Phase 1 - Hello CUDA (vector add)
- [ ] Phase 2 - Naive matmul
- [ ] Phase 3 - Tiled matmul with shared memory
- [ ] Phase 4 - Memory bandwidth
- [ ] Phase 5 - PCIe bandwidth
- [ ] Phase 6 - CLI tool with NVML telemetry
- [ ] Phase 7 - Stress mode

## References

- NVIDIA, "An Even Easier Introduction to CUDA"
- Kirk and Hwu, _Programming Massively Parallel Processors_
- NVIDIA CUDA C++ Best Practices Guide
- NVIDIA Nsight Compute documentation
- NVML developer reference
