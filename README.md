# gpu-stress-tester

A modern C++ and CUDA microbenchmark and stress-testing framework for GPUs.

This project is a hands-on portfolio piece for learning and demonstrating strong systems programming skills in modern C++ and CUDA. Its primary focus is building workloads that stress GPUs hard and verify they remain functional and correct under sustained heavy load — the kind of diagnostic and validation work done in real GPU systems and silicon validation environments.

Target hardware: NVIDIA RTX 3080 (Ampere, sm_86).

## Motivation

The goal is to build a stress and validation framework that can push GPUs to their limits while continuously checking for correctness drift, thermal throttling, and other signs of trouble. 

As a supporting activity, the project includes deliberate kernel optimization work (for example, taking a naive matrix multiply and turning it into a much faster tiled version). This work is not done purely for speed — it is used to develop the skills and techniques needed to write *efficiently stressful* kernels that can sustain very high memory or compute throughput.

In short: optimization is a tool in service of building better stress tests.

## What I've Learned

- **Modern C++ fundamentals** — RAII, smart pointers, `std::chrono`, exceptions, virtual interfaces, and clean resource management.
- **Benchmarking methodology** — Separating setup/teardown from timed work, collecting statistical results (mean + stddev), and writing deterministic verification.
- **Project infrastructure** — Structuring a real C++ project with CMake and clear separation between library code and executables.
- **CUDA programming patterns** — Kernel launch, memory hierarchy, shared memory, and optimization techniques — learned in the context of eventually applying them to create aggressive, sustained stress workloads.
- **Discipline of incremental delivery** — Working in well-defined phases with explicit acceptance criteria.

## Tech Stack & Tools

- **Language**: C++20
- **Build**: CMake 3.20+
- **Hardware**: NVIDIA RTX 3080 

## Build & Run

```bash
cmake -B build -S .
cmake --build build -j
./build/gpu-stress --benchmark cpu-matmul --size 256 --iterations 5
```

## Roadmap

The full detailed implementation plan lives in [plan.md](plan.md).

