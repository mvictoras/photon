# Photon

Production GPU path tracer built with **Kokkos** for portable execution across NVIDIA, AMD, and Intel GPUs. Implements a full **Disney Principled BSDF**, reads **pbrt-v4** scene files, and integrates with the **ANARI** rendering standard.

![Bathroom](renders/pbrt_bathroom_production.png)
*Bathroom scene — 592K triangles, 12 textures, 1920x1920, 1024 spp, OIDN denoised*

## Rendering Features

- **Wavefront path tracing** with configurable SPP and max bounces
- **Next Event Estimation (NEE)** with area light and emissive mesh sampling
- **Multiple Importance Sampling (MIS)** — power heuristic combining BSDF + light PDFs
- **Russian Roulette** — unbiased path termination
- **Adaptive sampling** — variance-based per-pixel early termination
- **Firefly clamping** — throughput limiting for noise reduction
- **ACES filmic tone mapping** with exposure control
- **Intel OIDN denoiser** integration

## Disney Principled BSDF

- Diffuse (Lambertian with Fresnel retro-reflection)
- Specular reflection (GGX microfacet with Smith geometry)
- Specular transmission (Fresnel-weighted thin-shell glass)
- Clearcoat (separate GGX lobe)
- Sheen (Ashikhmin model with tint)
- Anisotropic reflection (anisotropic GGX)
- Specular tint (colored dielectric reflections)
- Subsurface scattering (Christensen-Burley diffusion approximation)
- Conductor Fresnel (physical eta/k with named spectral data for Ag, Au, Cu, Al, Fe)

## Lighting

- Rectangular area lights with importance sampling
- Emissive triangle mesh sampling in NEE
- **HDR environment maps** with hierarchical importance sampling (PFM, HDR, PNG)
- One-sided emission (front-face only, matching pbrt-v4)

## Textures

- Runtime GPU texture sampling with bilinear filtering
- Flat texture atlas (single Kokkos::View) for multi-texture access
- Per-material texture support: base color, normal map, roughness, metallic, emission, alpha
- Image formats: TGA, PNG, JPEG, BMP, HDR, PFM

## Ray Tracing Backends

| Backend | Hardware | Status |
|---------|----------|--------|
| **Kokkos BVH** | Any (CUDA/HIP/SYCL/OpenMP) | Portable fallback — all shading on GPU, zero host sync |
| **OptiX 8** | NVIDIA GPU (RT cores) | **Fastest** — GPU-direct shading in raygen program, 7-29x faster than Kokkos BVH |
| **Embree 4** | Intel/AMD CPU | Working (CPU↔GPU transfer bottleneck) |
| **HIP RT** | AMD GPU | Stub (ready for implementation) |

Auto-selection: OptiX → Embree → Kokkos. Override with `PHOTON_BACKEND=kokkos|embree|optix`.

### Benchmarks (NVIDIA RTX 5000 Ada, 1024×1024, 128 spp)

| Scene | Triangles | Kokkos BVH | OptiX 8 | Speedup |
|-------|-----------|-----------|---------|---------|
| Cornell Box | 36 | 5.4s | 4.9s | 1.1x |
| Veach MIS | 3K | 1.2s | 0.4s | 3.2x |
| Spaceship | 457K | 55.8s | 5.7s | 9.8x |
| Bathroom | 592K | 28.4s | 1.8s | **16x** |
| Kitchen | 1.4M | 50.2s | 1.7s | **29x** |

## pbrt-v4 Scene Reader

Full parser for the pbrt-v4 scene format:

- `Film`, `Camera` (perspective with DOF), `Sampler`, `Integrator`
- `LookAt`, `Transform`, `Scale`, `Rotate`, `ConcatTransform`, `Identity`
- `MakeNamedMaterial` (diffuse, conductor, dielectric, coateddiffuse, diffusetransmission)
- `Texture` definitions with image file loading
- `Shape` (trianglemesh, plymesh, sphere)
- `AreaLightSource`, `LightSource` (infinite/environment)
- `Include`, `Import` (recursive file inclusion)
- `ObjectBegin/End/Instance` (geometry instancing with budget limits)
- `AttributeBegin/End` (scoped state)
- Named spectral data (metal-Ag-eta, metal-Au-k, etc.)
- PLY mesh loading (binary little-endian, ASCII)

## Quick Start

```bash
# 1. Build
./scripts/build.sh

# 2. Download test scenes
./scripts/download_scenes.sh scenes

# 3. Render
./build/app/photon_pbrt_render scenes/cornell-box/scene-v4.pbrt \
  -o cornell.ppm --spp 256 --width 1024 --height 1024 \
  --denoise --exposure 0.3
```

## Build

```bash
# Basic (Kokkos only, auto-detects CUDA)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# With pre-installed Kokkos (recommended for CUDA)
cmake -S . -B build -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos

# With Embree
cmake -S . -B build -DPHOTON_ENABLE_EMBREE=ON -Dembree_DIR=/path/to/embree

# With OptiX
cmake -S . -B build -DPHOTON_ENABLE_OPTIX=ON -DOptiX_INSTALL_DIR=/path/to/optix

# With OIDN denoiser
cmake -S . -B build -DPHOTON_ENABLE_OIDN=ON
```

## CLI Usage

```bash
photon_pbrt_render <scene.pbrt> [options]

Options:
  -o <file>           Output PPM file (default: pbrt_render.ppm)
  --width <W>         Image width (overrides scene)
  --height <H>        Image height (overrides scene)
  --spp <N>           Samples per pixel (overrides scene)
  --max-depth <N>     Max bounce depth (overrides scene)
  --exposure <F>      Exposure multiplier (default: 1.0)
  --denoise           Enable OIDN denoising
  --max-instances <N> Max instances per object (for large scenes)
  --max-triangles <N> Max total triangle budget
```

## ANARI Device

Photon also works as an ANARI device for integration with ANARI-compatible applications:

```bash
# Render ANARI test scenes
./build/app/photon_scene_render --device photon --scene test/instanced_cubes \
  --width 512 --height 512 --spp 64 --output render.ppm
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│              ANARI Device Layer                  │
│        PhotonDevice, SceneFromAnari              │
├──────────────────────────────────────────────────┤
│  pbrt-v4 Scene Reader    │    Texture Atlas      │
│  Parser, PLY, PFM, TGA   │    GPU Flat Buffer    │
├──────────────────────────────────────────────────┤
│       Wavefront Path Tracing Integrator          │
│  NEE, MIS, Adaptive Sampling, Disney BSDF        │
│  Firefly Clamp, ACES Tone Map, OIDN Denoise      │
│              (Kokkos — portable)                 │
├──────────────────────────────────────────────────┤
│            RayBackend Interface                  │
│         build_accel() / trace_rays()             │
├───────┬──────────┬──────────┬────────────────────┤
│Kokkos │  Embree  │  OptiX   │      HIP RT        │
│ BVH   │  (CPU)   │ (NVIDIA) │      (AMD)         │
└───────┴──────────┴──────────┴────────────────────┘
```

## Test Scenes

Download scenes from [Benedikt Bitterli's collection](https://benedikt-bitterli.me/resources/):

```bash
./scripts/download_scenes.sh scenes
```

| Scene | Triangles | Description |
|-------|-----------|-------------|
| cornell-box | 36 | Classic Cornell box with area light |
| spaceship | 457K | Spaceship with glass dome, metallic body |
| bathroom | 592K | Interior with textures, chrome, glass, mirror |
| living-room | 143K | HDRI environment lighting |
| staircase | 263K | Complex interior geometry |
| veach-mis | 3K | MIS test with sphere emitters |
| kitchen | 1.4M | Large interior, 90 materials |

The renderer also supports the [Disney Moana Island Scene](https://www.disneyanimation.com/resources/moana-island-scene/) (97M+ triangles).

## Dependencies

| Dependency | Version | Method | Required |
|-----------|---------|--------|----------|
| Kokkos | 5.0+ | find_package / FetchContent | Yes |
| ANARI SDK | 0.15.0 | FetchContent | Yes |
| Embree | 4.x | find_package | Optional |
| OptiX | 8.x | find_package | Optional |
| HIP RT | ROCm 5.x+ | find_package | Optional |
| Intel OIDN | 2.x | find_package | Optional |
| stb_image | (bundled) | via ANARI SDK | Yes |

## License

See individual source files for license information.
