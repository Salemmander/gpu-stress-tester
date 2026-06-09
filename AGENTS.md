# Repository Guidelines

## Project Structure & Module Organization

This is a C++20/CUDA GPU benchmark and stress-testing project. Public interfaces
live in `include/`, implementations live in `src/`, and the CLI entrypoint is
`src/main.cpp`. Benchmark classes follow the existing `Benchmark` interface:
`setup()`, `run()`, `verify()`, and `teardown()`.

Key files:

- `include/Benchmark.h`: shared benchmark interface.
- `include/BenchmarkRunner.h` and `src/BenchmarkRunner.cpp`: timing loop and
  mean/stddev reporting.
- `src/*Benchmark.cpp` and `src/*Benchmark.cu`: CPU and CUDA benchmark
  implementations.
- `profile_*.ncu-rep` and `*.csv`: profiling outputs and analysis data.

## Build, Test, and Development Commands

- `cmake --preset dev`: configure the project into `build/` using the dev
  preset and `/usr/bin/g++-15` as the CUDA host compiler.
- `cmake -B build -S .`: configure without presets.
- `cmake --build build -j`: compile `gpu-stress-lib` and the `gpu-stress`
  executable.
- `./build/gpu-stress --benchmark cpu-matmul --size 256 --iterations 5`: run a
  CPU correctness/performance smoke test.
- `./build/gpu-stress --benchmark gpu-matmul-tiled --size 2048 --iterations 5`:
  run the tiled CUDA matrix multiply benchmark.

## Coding Style & Naming Conventions

Use modern C++20 with explicit ownership and RAII. Keep includes at the top of
files. Prefer readable, direct code over premature templates. Add comments only
for non-obvious CUDA or benchmarking decisions.

Follow the existing naming style: classes use `PascalCase`
(`GpuMatmulTiledBenchmark`), files mirror class names, and CUDA source files use
`.cu`. The build enables `-Wall -Wextra -Wpedantic`; keep new code warning-free.

## Testing Guidelines

There is currently no standalone test framework. Treat benchmark `verify()`
methods as the correctness gate. For CUDA kernels, test a small size first, then
the target stress size, for example `256` before `2048`. Keep setup and
verification outside the timed `run()` path.

## Commit & Pull Request Guidelines

Recent commits use short imperative or descriptive messages, such as `Add tiled
GPU matmul benchmark with shared memory` and `Complete Phase 2: Naive GPU matmul
+ TFLOPS reporting`. Keep commits focused on one benchmark, infrastructure
change, or profiling update.

Pull requests should include the benchmark changed, commands run, representative
timing results, and any profiler evidence when CUDA performance changes. Mention
target hardware assumptions, especially RTX 3080 / Ampere-specific tuning.

## Security & Configuration Tips

Do not commit generated build outputs from `build/`. Treat Nsight Compute
reports and CSV profiling data as local artifacts unless they document a
specific performance result. Avoid hard-coding machine-specific paths beyond
existing CMake preset configuration.
