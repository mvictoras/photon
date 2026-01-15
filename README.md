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
ctest --test-dir build -R '^opencode_'

## Run (standalone render)
```bash
./build/app/opencode_pt_render
open out.ppm
```

Current scene: a simple triangle-mesh quad rendered via BVH.

## Run (ANARI load smoke)
```bash
DYLD_LIBRARY_PATH=build/src:build/_deps/anari-build ./build/app/opencode_anari_smoke
```
```

## Status
Early scaffolding.
