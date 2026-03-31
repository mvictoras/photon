# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Photon** is a production GPU path tracer built with Kokkos for portable execution across NVIDIA/AMD/Intel GPUs. It implements the full Disney Principled BSDF, reads pbrt-v4 scenes, and exposes an ANARI rendering device.

## Build

```bash
# Basic build (reads KOKKOS_DIR, EMBREE_DIR, OPTIX_DIR from env)
./scripts/build.sh

# Manual CMake (OIDN is always enabled by build.sh)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos \
  -DPHOTON_ENABLE_OPTIX=ON -DOptiX_INSTALL_DIR=/path/to/optix \
  -DPHOTON_ENABLE_EMBREE=ON -Dembree_DIR=/path/to/embree \
  -DPHOTON_ENABLE_OIDN=ON
cmake --build build -j$(nproc)
```

The main option flag is `OPENCODE_ENABLE_CUDA` (ON by default). When CUDA is enabled, CMake uses its native CUDA language support to compile `.cpp` photon sources through nvcc. ANARI SDK files are explicitly excluded from CUDA compilation to avoid `float4` conflicts with linalg.

## Running

The binary is `./build/app/photon_pbrt_render` (or `./build/_deps/icet-build/bin/photon_pbrt_render` when MPI/IceT is enabled).

```bash
# Set up environment (local install paths)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/home/mvictoras/src/photon/build:/home/mvictoras/src/embree-install/lib:/home/mvictoras/src/kokkos-cuda-install/lib:$LD_LIBRARY_PATH

# Render a scene (backend auto-selects: OptiX → Embree → Kokkos)
PHOTON_BACKEND=optix ./build/app/photon_pbrt_render scenes/bathroom/scene-v4.pbrt \
  -o renders/out.ppm --spp 64 --width 1024 --height 1024

# With denoising
PHOTON_BACKEND=optix ./build/app/photon_pbrt_render scenes/bathroom/scene-v4.pbrt \
  -o renders/out.ppm --spp 1024 --denoise --exposure 0.3

# Large instanced scenes (Moana)
PHOTON_BACKEND=optix ./build/app/photon_pbrt_render scene.pbrt \
  -o out.ppm --spp 16 --use-ias --max-instances 50 --max-triangles 5000000
```

CLI flags: `-o`, `--width`, `--height`, `--spp`, `--max-depth`, `--exposure`, `--denoise`, `--use-ias`, `--max-instances <N>`, `--max-triangles <N>`.

## Tests

**Important:** Tests are skipped when Kokkos CUDA is enabled (`OPENCODE_ENABLE_CUDA=ON`) because CUDA CXX propagation is incompatible with the test build. To run tests, build without CUDA:

```bash
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DOPENCODE_ENABLE_CUDA=OFF
cmake --build build-tests -j$(nproc)
cd build-tests && ctest

# Single test
./build-tests/photon_math_test
./build-tests/photon_disney_bsdf_test
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│              ANARI Device Layer                  │
│  src/photon/anari/ — PhotonDevice, SceneFromAnari│
├──────────────────────────────────────────────────┤
│  pbrt-v4 Reader (src/photon/pbrt/)               │
│  Parser → pbrt_to_photon → Scene                 │
├──────────────────────────────────────────────────┤
│  Wavefront Path Tracer (src/photon/pt/)          │
│  NEE, MIS, Adaptive Sampling, Disney BSDF        │
│  OIDN denoiser  ·  ACES tone map                 │
│         (all Kokkos — runs on any GPU)           │
├──────────────────────────────────────────────────┤
│  RayBackend interface (build_accel / trace_*)    │
├──────┬──────────┬──────────┬─────────────────────┤
│Kokkos│  Embree  │  OptiX 8 │  HIP RT (stub)      │
│ BVH  │  (CPU)   │ (NVIDIA) │                     │
└──────┴──────────┴──────────┴─────────────────────┘
```

### Key abstractions

**`RayBackend`** (`include/photon/pt/backend/ray_backend.h`) — pure virtual interface: `build_accel(Scene&)`, `trace_closest(RayBatch&, HitBatch&)`, `trace_occluded(RayBatch&, View<u32*>)`. Selected at runtime via `BackendFactory` using the `PHOTON_BACKEND` env var; auto-priority is OptiX → Embree → Kokkos.

**`Scene`** (`include/photon/pt/scene.h`) — flat GPU-friendly struct: `TriangleMesh`, `Bvh`, `Kokkos::View<Material*>`, `TextureAtlas` (single flat view), `Kokkos::View<Light*>`, optional `EnvironmentMap`, and `emissive_prim_ids` for NEE. Instanced scenes (Moana) use a two-level IAS built in `optix_backend`.

**`PathTracer`** (`include/photon/pt/pathtracer.h`) — takes a `Scene` and `RayBackend`, returns a `RenderResult` with color/depth/normal/albedo AOVs.

**`EnvironmentMap`** (`include/photon/pt/environment_map.h`) — hierarchical importance-sampled HDR map with conditional/marginal CDFs and a rotation matrix for world↔texture transforms.

**`Material`** (`include/photon/pt/material.h`) — full Disney Principled parameters + per-channel texture indices into the `TextureAtlas`.

### Data flow: pbrt scene load

`pbrt_parser.cpp` tokenizes and builds a `PbrtScene` (intermediate representation) → `pbrt_to_photon.cpp` converts it to a `Scene` (flattening instances, building BVH, uploading textures to GPU via Kokkos::View).

### Portable GPU code conventions

All shading code uses `KOKKOS_FUNCTION` / `KOKKOS_LAMBDA` to compile on both host and device. GPU data lives in `Kokkos::View<T*>` (equivalent to device pointers with ref-counting). Never use raw `new`/`delete` for GPU data.
