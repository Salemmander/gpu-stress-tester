# gpu-stress-tester

A modern C++ and CUDA microbenchmark and stress-testing framework for GPUs.

This project is a hands-on portfolio piece for learning and demonstrating strong systems programming skills in modern C++ and CUDA. It focuses on building diagnostic-style workloads that exercise compute units, memory subsystems, and data movement paths while producing measurable performance results.

Target hardware: NVIDIA RTX 3080 (Ampere, sm_86).

## Motivation

The goal is to build tools that push GPUs hard and report what comes back — similar to the kind of validation and stress workloads used in real GPU silicon validation. The project serves as both deep technical practice and a demonstration of solid engineering habits: clean interfaces, testing discipline, benchmarking methodology, and incremental delivery.

## What I've Learned

- **Modern C++ fundamentals** — RAII, smart pointers, `std::chrono`, exceptions, virtual interfaces, and clean resource management.
- **Testing & design for testability** — Writing an abstract benchmark interface, using test doubles (spy pattern), and using `friend` classes to enable strong correctness tests without polluting the public API.
- **Benchmarking methodology** — Separating setup/teardown from timed work, collecting statistical results (mean + stddev), and writing deterministic verification.
- **Project infrastructure** — Structuring a real C++ project with CMake, FetchContent, GoogleTest, and clear separation between library code, executables, and tests.
- **Discipline of incremental delivery** — Working in well-defined phases with explicit acceptance criteria.

## Tech Stack & Tools

- **Language**: C++20
- **Build**: CMake 3.20+
- **Testing**: GoogleTest (via FetchContent)
- **Target**: NVIDIA RTX 3080 (future phases will use CUDA and NVML)

## Build & Run

```bash
cmake -B build -S .
cmake --build build -j
./build/gpu-stress --benchmark cpu-matmul --size 256 --iterations 5
ctest --test-dir build
```

## Roadmap

The full detailed implementation plan lives in [plan.md](plan.md).

## References

- NVIDIA CUDA documentation and best practices guides
- Kirk & Hwu — *Programming Massively Parallel Processors*
- NVIDIA Nsight Compute and NVML documentation
