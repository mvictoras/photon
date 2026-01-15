# opencode-pathtracer

ANARI device: production path tracer built with **C++23** and **Kokkos** (CUDA/HIP/SYCL).

## Goals
- ANARI SDK v0.15.0 compatible device
- Multi-backend GPU execution via Kokkos
- Modern path tracing features (MIS, HDR env, BVH, next-event estimation, etc.)
- Unit tests + CI-friendly tooling

## Build (dev)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Status
Early scaffolding.
