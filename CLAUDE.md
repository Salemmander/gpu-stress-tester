# gpu-stress-tester — agent context

A CUDA-based GPU stress tester and microbenchmark suite written in modern C++,
built as a hands-on portfolio piece for the NVIDIA System Software Engineer - GPU
role (req JR2016824). The phased plan in `plan.md` is the canonical roadmap.

Target hardware: NVIDIA RTX 3080 (Ampere, sm_86, 10GB GDDR6X, PCIe Gen4 x16).

## Why this project exists (context for design choices)

The role builds diagnostic / stress / validation tools for GPU silicon using
C++, CUDA, and Vulkan. The job emphasis is "the most stressful set of
applications a GPU or HPC server would see in its life cycle."

This project deliberately:

- Starts CPU-only in Phase 0 to demonstrate modern C++ fundamentals (RAII,
  smart pointers, templates, `std::chrono`, exceptions, CMake)
  before any GPU code is introduced.
- Writes naive kernels first (Phase 2), then optimizes (Phase 3+), so the
  speedup story and profiling discipline are visible in the git history.
- Reports measured performance against theoretical peaks (FLOPS, GB/s, PCIe).

The intended language ratio across the project is roughly 70% modern C++ /
30% CUDA. C++ is primary; CUDA is the GPU-specific tool.

## Author profile

Salem Nassar — Software Engineer at Verizon (~2 yrs YoE), BS CS UT Dallas (2024),
MS CS Georgia Tech online (in progress, ML specialization).

- Strong: Python, Kubernetes/OpenShift/Docker/Helm, GitLab CI/CD, PostgreSQL.
- Learning here: deep modern C++, OS fundamentals, computer architecture, CUDA.
- Prior C++ exposure is shallow (one YOLOv11 + PID controller side project).

Implication: favor simple, idiomatic minimum-viable code over clever generality
or premature abstractions. Three repeated lines beats a half-finished template.

## Conventions

### Language and build

- C++20.
- Warnings: `-Wall -Wextra -Wpedantic`. Treat warnings as errors in CI later.
- Release: `-O2`. Debug: `-O0 -g`.
- CMake 3.20+ minimum.
- No test framework is used (tests were removed).
- No CUDA dependencies (`find_package(CUDAToolkit)`, `.cu` files, `nvcc`) enter
  the build until Phase 1. Phase 0 is pure host C++.

### Directory layout (target for Phase 0)

```
gpu-stress-tester/
├── CMakeLists.txt
├── include/
│   ├── Benchmark.h              # abstract base class
│   ├── BenchmarkRunner.h        # times runs, computes mean and stddev
│   └── CpuMatmulBenchmark.h
├── src/
│   ├── main.cpp                 # CLI entrypoint
│   ├── BenchmarkRunner.cpp
│   └── CpuMatmulBenchmark.cpp
```

### `Benchmark` interface (agreed shape)

Abstract base class with the following virtual methods, called in order by
`BenchmarkRunner`:

- `name()` — human-readable benchmark name
- `setup()` — allocate buffers, prepare inputs (not timed)
- `run()` — the timed work
- `verify()` — correctness check after `run()`; throws on mismatch
- `teardown()` — release resources (also handled by RAII; `teardown()` is for
  anything that cannot be cleanly tied to a destructor)

`BenchmarkRunner` repeats the `setup → run → verify` cycle N times, times only
`run()` with `std::chrono::steady_clock`, and reports mean and stddev.

### CLI

Hand-rolled argv parsing for Phase 0 (`--benchmark`, `--size`, `--iterations`).
Pulling in `cxxopts` is deferred to Phase 6 when the CLI gets polished and JSON
output lands.

## Working preferences

- No emojis in any files.
- Imports / includes at the top of the file, never inside functions.
- Confirm the approach before editing files, especially for multi-file changes.
- Show raw command output. Do not summarize unless asked.
- Focus on root cause. No workarounds unless root cause is confirmed out of scope.
- Concise by default. Full detail when explaining "why" or "how" something works.
- Default to no comments. Only add a comment when the WHY is non-obvious.

## Current status

- Initial commit (`README.md`, `.gitignore`) is on `master`, pushed to
  `origin` (private repo `Salemmander/gpu-stress-tester`).
- Phase 0 design is approved (see below). Scaffolding has not been written yet.

## Phase 0 plan (approved, ready to implement)

Files to create:

- `CMakeLists.txt` — top level. C++20, builds the `gpu-stress` binary.
- `include/Benchmark.h` — ABC as described above.
- `include/BenchmarkRunner.h` + `src/BenchmarkRunner.cpp` — runs a benchmark
  N times, computes mean and stddev, prints a small report.
- `include/CpuMatmulBenchmark.h` + `src/CpuMatmulBenchmark.cpp` — naive triple
  loop `C = A * B` for square matrices, configurable size, `verify()` against
  a slow reference or known small input.
- `src/main.cpp` — argv parsing, dispatches to the requested benchmark.

Phase 0 acceptance bar:

1. `cmake -B build -S . && cmake --build build -j` succeeds with no warnings.
2. `./build/gpu-stress --benchmark cpu-matmul --size 256 --iterations 5`
   runs end-to-end and prints a timing report with mean and stddev.

When all three pass, Phase 0 is done and Phase 1 (Hello CUDA: vector add wired
into the framework) can begin.


## Session Handoff — 2026-05-25 (Phase 3 progress)

**Current Project State**
- Phase 2 (Naive matmul): Complete. `GpuMatmulBenchmark` reaches ~1.26 TFLOPS at n=2048.
- Phase 3 (Tiled matmul): In active development. `GpuMatmulTiledBenchmark` implemented with shared memory tiling.

**Recent Work Completed**
- Full implementation of tiled kernel (`matmul_tiled`) with cooperative tile loading into shared memory (`As`/`Bs`), `__syncthreads()`, and inner compute loop.
- Separate class `GpuMatmulTiledBenchmark` created (header + .cu).
- Refactored benchmark dispatch in `main.cpp` from long if-else chain into a clean `std::map` called `benchmarks` (much easier to add new kernels).
- Added `CMakePresets.json` (minimal dev preset with g++-15).
- Removed entire `tests/` directory and all GoogleTest integration (per user preference).
- Extensive Nsight Compute profiling (both CLI and GUI) comparing naive vs tiled versions.
- Updated CLAUDE.md, plan.md, and README.md to reflect removal of tests and current state.
- Cleaned `.gitignore` significantly.

**Key Findings from Profiling (n=2048)**
- Naive: ~1.26 TFLOPS
- Tiled TILE_SIZE=16 (no padding): Best result so far at ~1.51 TFLOPS
- Tiled TILE_SIZE=32: Regressed to ~1.02 TFLOPS with high variance (stddev often 7-8ms on ~17-19ms runs)
- Massive win on memory traffic: Global loads dropped ~94% with tiling (Nsight confirms this clearly).
- However, L1 hit rate dropped significantly (87% → ~51%).
- At 32x32: Very low occupancy — theoretical and achieved active warps both ~33%.
- Millions of shared memory bank conflicts reported at 32 (4M+ in one launch).
- Padding (+1 and +4) has not improved wall time (and sometimes made things slightly worse, likely due to occupancy).

**Current Blockers / Issues**
- TILE_SIZE=32 is slower than 16 despite much better memory traffic numbers. Root cause appears to be occupancy collapse + shared memory pressure, not just bank conflicts.
- High timing variance on larger tile sizes.
- Still requires `sudo` for `ncu` (permission issue on this machine; modprobe fix attempted but not resolved).
- CLAUDE.md had duplicated old handoff text appended at one point (cleaned up).

**Next Steps (recommended order)**
1. Decide whether to double down on fixing TILE_SIZE=32 (try rectangular blocks like 32x16, better padding strategy, occupancy tuning) or go back and polish the 16 version.
2. Use Nsight Compute reports (we have `profile_2048.ncu-rep` and `profile_tiled32_2048.ncu-rep`) to compare occupancy, shared memory throughput, and bank conflicts in detail.
3. Once a clearly superior tiled version exists, run head-to-head Nsight comparisons with the same metrics file.
4. Consider trying other tile sizes or launch configurations.

**Files of Interest**
- `src/GpuMatmulTiledBenchmark.cu` (main kernel + class)
- `src/main.cpp` (now uses clean `benchmarks` map)
- `metrics.txt` (current profiling metrics)
- `profile_*.ncu-rep` files in root (for GUI analysis)

**Working Preferences (unchanged)**
- User prefers guided, section-by-section help when writing CUDA code.
- Likes to write most of the code themselves.
- Prefers minimal, readable code over heavy abstractions when possible.
- Using `sudo ncu` as temporary workaround for permission issues.

