# photon

ANARI device: production path tracer (**photon**) built with **C++23** and **Kokkos** (CUDA/HIP/SYCL).

## Goals
- ANARI SDK v0.15.0 compatible device
- Multi-backend GPU execution via Kokkos
- Modern path tracing features (MIS, HDR env, BVH, next-event estimation, etc.)
- Unit tests + CI-friendly tooling

## Build (dev)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -R '^photon_'

## Run (standalone render)
```bash
./build/app/photon_render
open out.ppm
```

Current scene: two quads (one vertical, one ground) rendered via BVH.

## Run (ANARI load smoke)
```bash
DYLD_LIBRARY_PATH=build/src:build/_deps/anari-build ./build/app/photon_anari_smoke
```
```

## Status
Early scaffolding.
