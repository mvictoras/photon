# photon

Production path tracer built as an **ANARI** device with **C++23** and **Kokkos** for portable GPU execution (CUDA/HIP/SYCL).

## Features

### Rendering
- **Wavefront path tracing** integrator with configurable SPP and max bounces
- **Next Event Estimation (NEE)** — direct light sampling for fast convergence
- **Multiple Importance Sampling (MIS)** — power heuristic combining BSDF + light PDFs
- **Russian Roulette** — unbiased path termination after bounce 2
- **Disney Principled BSDF** — metallic, roughness, transmission, clearcoat, subsurface scattering
- **GGX microfacet model** with Fresnel and Smith geometry term

### Lighting
- Point, directional, spot, and area lights
- **HDR environment maps** with hierarchical importance sampling
- Emissive geometry (any mesh can be a light source)

### Geometry & Acceleration
- Triangle and quad mesh support with smooth normals and UVs
- **BVH acceleration** (CPU-built, median split)
- **Pluggable ray-tracing backends**:
  - **Kokkos BVH** (portable fallback — works everywhere)
  - **Embree 4** (Intel CPU — SSE/AVX/AVX-512)
  - **OptiX 7+** (NVIDIA GPU — RT cores)
  - **HIP RT** (AMD GPU — RDNA2+)
- **Automatic backend selection** based on available hardware
- Override via `PHOTON_BACKEND=embree|optix|hiprt|kokkos`

### Volumetric Rendering
- Heterogeneous volumes with grid-based density
- **Delta tracking** (Woodcock tracking) for unbiased free-flight sampling
- **Henyey-Greenstein** phase function for anisotropic scattering

### Camera
- Perspective with configurable FOV
- **Thin lens depth of field** (aperture + focus distance)
- Orthographic projection

### Post-Processing
- **ACES filmic** and **Reinhard** tone mapping
- **Intel OIDN** denoiser integration (optional)
- **AOV channels**: color, depth, normal, albedo
- sRGB gamma correction

### ANARI Device
- ANARI SDK v0.15.0 compatible device
- Camera, Material (physicallyBased), Light, Geometry, Surface, World objects
- Frame rendering with multiple output channels

### Textures
- 2D image textures with bilinear filtering and UV wrapping
- Texture atlas for GPU-efficient multi-texture access
- Per-material texture assignments (base color, normal, roughness, metallic, emission)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build -j
ctest --test-dir build -R '^photon_'
```

### Optional backends

```bash
# Embree (auto-detected if installed)
cmake -S . -B build -DPHOTON_ENABLE_EMBREE=ON

# OptiX (requires NVIDIA GPU + OptiX SDK)
cmake -S . -B build -DPHOTON_ENABLE_OPTIX=ON -DOptiX_INSTALL_DIR=/path/to/optix

# HIP RT (requires AMD GPU + ROCm)
cmake -S . -B build -DPHOTON_ENABLE_HIPRT=ON

# Kokkos GPU backends
cmake -S . -B build -DOPENCODE_ENABLE_CUDA=ON   # NVIDIA
cmake -S . -B build -DOPENCODE_ENABLE_HIP=ON    # AMD
cmake -S . -B build -DOPENCODE_ENABLE_SYCL=ON   # Intel

# Intel OIDN denoiser (auto-detected if installed)
cmake -S . -B build -DPHOTON_ENABLE_OIDN=ON
```

## Run

```bash
# Standalone render (Cornell box → out.ppm)
./build/app/photon_render
open out.ppm

# ANARI device render
LD_LIBRARY_PATH=build/src:build/_deps/anari-build ./build/app/photon_anari_render
open anari_out.ppm
```

## Architecture

```
┌─────────────────────────────────────────────┐
│            ANARI Device Layer               │
│     PhotonDevice, SceneFromAnari            │
├─────────────────────────────────────────────┤
│      Wavefront Path Tracing Integrator      │
│    NEE, MIS, Russian Roulette, Disney BSDF  │
│         (Kokkos — portable)                 │
├─────────────────────────────────────────────┤
│           RayBackend Interface              │
│        build_accel() / trace_rays()         │
├──────┬──────────┬──────────┬────────────────┤
│Kokkos│  Embree  │  OptiX   │    HIP RT      │
│ BVH  │  (CPU)   │ (NVIDIA) │    (AMD)       │
└──────┴──────────┴──────────┴────────────────┘
```

## Tests

15 unit/integration tests covering math, sampling, materials, lighting, textures, camera, BVH, backend, path tracer, volumes, tone mapping, and full pipeline integration.

```bash
ctest --test-dir build -R '^photon_' --output-on-failure
```

## Dependencies

| Dependency | Version | Method | Required |
|-----------|---------|--------|----------|
| Kokkos | 5.0.1 | FetchContent | Yes |
| ANARI SDK | 0.15.0 | FetchContent | Yes |
| Embree | 4.x | find_package | Optional |
| OptiX | 7.x+ | find_package | Optional |
| HIP RT | ROCm 5.x+ | find_package | Optional |
| Intel OIDN | 2.x | find_package | Optional |
