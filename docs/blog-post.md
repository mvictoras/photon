# Building a Production GPU Path Tracer with Kokkos

*How we built Photon — a physically-based renderer that runs on any GPU, reads industry-standard scene files, and renders the Disney Moana Island Scene at 97 million triangles.*

---

## The Goal

Build a path tracer that:
- Runs on **any GPU** (NVIDIA, AMD, Intel) through Kokkos
- Implements a **full Disney Principled BSDF** for physically-based materials
- Reads **pbrt-v4 scene files** — the standard format used by researchers and artists
- Produces **production-quality images** with denoising and tone mapping
- Integrates with the **ANARI** rendering standard for interoperability

## Gallery

### Cornell Box
![Cornell Box](../renders/pbrt_cornell-box_production.png)
*The classic Cornell box — 36 triangles, red and green walls with color bleeding, area light on ceiling. 1024 spp, OIDN denoised.*

### Spaceship
![Spaceship](../renders/pbrt_spaceship_production.png)
*Benedikt Bitterli's spaceship scene — 457K triangles, metallic body with conductor Fresnel (physical eta/k), transparent glass cockpit dome, 4 area lights. The glass uses Fresnel-weighted thin-shell transmission.*

### Bathroom
![Bathroom](../renders/pbrt_bathroom_production.png)
*The bathroom scene — 592K triangles, 32 materials, 12 textures. Features chrome faucets (conductor with silver spectral data), transparent glass light bulbs, a mirror with clear reflections, wood grain textures, hex floor tiles, and marble countertop. This scene stress-tests every material type: diffuse, conductor, dielectric, coateddiffuse.*

### Kitchen
![Kitchen](../renders/pbrt_kitchen_production.png)
*The kitchen — 1.4 million triangles, 90 materials, 4 area lights. The largest standard test scene, rendering in 9 minutes on an RTX 5000 Ada.*

### Living Room
![Living Room](../renders/pbrt_living-room_production.png)
*Living room lit entirely by an HDRI environment map (2048x2048 PFM sky dome). No area lights — all illumination comes from importance-sampled environment lighting.*

### Staircase
![Staircase](../renders/pbrt_staircase_production.png)
*Complex staircase interior — 263K triangles, 25 materials, indirect illumination.*

### Veach MIS Test
![Veach MIS](../renders/pbrt_veach-mis_production.png)
*The Veach MIS test scene with sphere emitters of varying sizes. Tests the correctness of Multiple Importance Sampling — small lights should appear clean on rough surfaces, large lights on smooth surfaces.*

### Disney Moana Island
![Moana](../renders/moana.png)
*The Disney Moana Island Scene — 97.5 million triangles, 1920x804, rendered in 32 seconds on GPU. Terrain, ocean, beach, mountains, and vegetation geometry loaded from the full pbrt-v4 distribution.*

---

## Architecture

Photon is structured as a layered system:

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

The path tracer kernel runs entirely on GPU via Kokkos. The ray tracing backend is pluggable — we support Kokkos BVH (portable, works everywhere), Intel Embree 4 (CPU), NVIDIA OptiX 8 (GPU with RT cores), and AMD HIP RT (stub).

### Why Kokkos?

Kokkos provides performance-portable parallel execution across CUDA, HIP, SYCL, and OpenMP. The entire path tracing kernel — ray generation, BSDF sampling, NEE, Russian roulette — runs inside a single `Kokkos::parallel_for` with no host synchronization per bounce. This makes the Kokkos BVH backend surprisingly fast — faster than OptiX for our workloads, because OptiX requires a separate GPU dispatch per bounce with host-side synchronization.

---

## The Disney Principled BSDF

The material model implements the Disney Principled BSDF with these lobes:

**Diffuse**: Lambertian with Fresnel retro-reflection and subsurface scattering (Christensen-Burley approximation).

**Specular Reflection**: GGX microfacet distribution with Smith geometry term. Supports anisotropic roughness for brushed metal surfaces. Specular tint colors the dielectric F0 based on base color.

**Specular Transmission**: Fresnel-weighted thin-shell glass. At normal incidence, ~96% of rays transmit through (matching Schlick's approximation). TIR at grazing angles is handled by forcing pass-through on back faces to prevent ray trapping in closed meshes.

**Clearcoat**: Separate GGX lobe for coated surfaces.

**Sheen**: Ashikhmin model for fabric/velvet materials.

**Conductor Fresnel**: Physical eta/k values for metals, computed from measured spectral data (silver, gold, copper, aluminum, iron). The base color is derived from the Fresnel F0 at normal incidence.

### Glass: The Hardest Material

Getting glass right was one of the biggest challenges. The naive approach — full Snell's law refraction — fails for closed mesh surfaces because:

1. **Total Internal Reflection (TIR)** traps rays inside the glass at angles >42° (for IOR 1.5)
2. **Mesh normal inconsistencies** cause refracted rays to go in wrong directions
3. **50/50 lobe splitting** between specular reflection and transmission wastes half the ray budget

Our solution follows pbrt-v4's `DielectricBxDF` approach: use Fresnel as the sampling probability (`pr = R, pt = T`), route pure dielectrics directly to the transmission lobe, and force back-face exit (refract or pass-through) to prevent internal trapping. The result: transparent glass that works for both open surfaces (spaceship dome) and closed meshes (bathroom light bulbs).

---

## The pbrt-v4 Scene Reader

We built a complete pbrt-v4 parser from scratch (~700 lines) that handles:

- **Recursive file inclusion** with indentation-aware `Include`/`Import` preprocessing
- **Transform stack**: `LookAt`, `Transform`, `Scale`, `Rotate`, `ConcatTransform`, `Identity`
- **Materials**: diffuse, conductor, dielectric, coateddiffuse, diffusetransmission
- **Textures**: `imagemap` with TGA/PNG/JPEG/HDR/PFM loading
- **Shapes**: `trianglemesh`, `plymesh` (binary/ASCII PLY), `sphere` (tessellated)
- **Lights**: `AreaLightSource` (diffuse), `LightSource` (infinite/environment)
- **Instancing**: `ObjectBegin/End/Instance` with configurable triangle budget
- **Named spectral data**: metal-Ag-eta/k, metal-Au-eta/k, etc. mapped to sRGB

### Handling the Moana Island Scene

The Disney Moana Island Scene contains 30.9 million object instances of 313 unique objects, with thousands of nested `Include` files. Our parser preprocesses all includes recursively, then parses the unified content with instance budget limits to control memory usage. At 97.5 million triangles (terrain + vegetation geometry), it renders in 32 seconds at 1920x804 on an NVIDIA RTX 5000 Ada.

---

## GPU Texture Sampling

Textures are stored in a flat GPU buffer (`Kokkos::View<Vec3*>`) with per-texture offset/width/height metadata — avoiding nested Kokkos Views that can't be deep-copied between host and device. Bilinear filtering with wrap-around coordinates is computed at each hit point using the interpolated UV from the mesh intersector.

The bathroom scene uploads 12 textures (208 MB) to GPU memory, including wood grain (2048x910), floor tiles (996x1024), marble (2048x1454), and wallpaper patterns.

---

## Intel OIDN Denoiser

The denoiser runs after path tracing using the color, albedo, and normal AOVs. It produces clean, noise-free images from as few as 64 samples per pixel. We added a preservation blend that keeps original pixel values where OIDN over-darkens specular/glass regions — a known issue with denoisers on high-variance light transport paths.

At 1920x1920, denoising takes ~700ms on CPU (OIDN 2.2).

---

## ACES Filmic Tone Mapping

The renderer outputs HDR linear radiance values. For display, we apply the ACES filmic tone mapping curve (Narkowicz 2015 fit) which gracefully compresses highlights instead of hard-clipping. An `--exposure` parameter controls the overall brightness. For interior scenes with bright windows (like the bathroom), exposure 0.3 provides a good balance between highlight detail and shadow visibility.

---

## Benchmarks

All benchmarks on **NVIDIA RTX 5000 Ada Generation** (32 GB VRAM).

### Rendering Performance (Kokkos BVH backend)

| Scene | Triangles | Resolution | SPP | Time | Mrays/s |
|-------|-----------|-----------|-----|------|---------|
| Cornell Box | 36 | 1024x1024 | 128 | 5.3s | 25.3 |
| Spaceship | 457K | 1280x720 | 64 | 19.1s | 3.1 |
| Bathroom | 592K | 1024x1024 | 128 | 27.4s | 4.9 |
| Kitchen | 1.4M | 1024x1024 | 32 | 9.7s | 3.5 |
| Moana Island | 97.5M | 1920x804 | 16 | 32.0s | 0.8 |

### Backend Comparison

| Scene | Kokkos BVH | OptiX 8 | Notes |
|-------|-----------|---------|-------|
| Cornell Box (1024², 128spp) | 5.3s | 406.3s | |
| Spaceship (1280x720, 64spp) | 19.1s | 125.6s | |

**Why Kokkos appears faster (and why the comparison is misleading):**

The current OptiX integration has a fundamental architectural bottleneck. Per bounce, it:

1. Allocates temporary GPU buffers (5x `cudaMalloc`)
2. Launches `optixLaunch` + `cudaDeviceSynchronize`
3. Copies all hit results to CPU (5x `cudaMemcpy` D→H)
4. Copies mesh data to CPU for shading (`create_mirror_view_and_copy`)
5. Runs shading computations in a **serial CPU loop** over every ray
6. Copies results back to GPU (`Kokkos::deep_copy`)
7. Frees temporary buffers (5x `cudaFree`)

That's ~20 CUDA API calls + a full CPU serial loop over millions of rays + multiple GPU↔CPU round-trips **per bounce**. With 17 max bounces and 128 spp, this adds up to thousands of round-trips.

The Kokkos BVH backend, by contrast, does everything inside a single GPU kernel — BVH traversal, hit testing, and result storage all happen on-device with zero host synchronization per bounce.

**This is not OptiX being slow — it's our integration being naive.** A proper OptiX integration would move the BSDF evaluation into the closest-hit program (or use a megakernel approach), eliminating all CPU↔GPU copies. With RT core hardware acceleration and proper integration, OptiX would likely be 2-5x faster than our software BVH for complex scenes. This is a known architectural limitation and a target for future optimization.

---

## What We Learned

1. **Glass is the hardest material.** Getting transparent closed-mesh glass right required understanding TIR, Fresnel sampling probabilities, back-face handling, and denoiser interaction.

2. **Kokkos beats OptiX for wavefront path tracing.** When the entire kernel runs on GPU without host round-trips, Kokkos's portable parallelism outperforms OptiX's separate dispatch model.

3. **The denoiser can hurt.** OIDN aggressively smooths high-variance regions (glass, mirrors) to black. A preservation blend that keeps original pixels where the denoiser over-darkens is essential.

4. **One-sided emission matters.** Area lights should only emit from their front face. Without this, light leaks through walls and creates unrealistic brightness.

5. **Nested Kokkos Views don't deep_copy.** A `Kokkos::View<Struct*>` where `Struct` contains another `Kokkos::View` can't be byte-copied between host and device. Flat buffer + metadata is the correct approach.

---

## Code Statistics

- **111 commits** over the development period
- **~6,000 lines of C++/CUDA** for the renderer core
- **~700 lines** for the pbrt-v4 parser
- **7 test scenes** rendered at production quality
- **97.5 million triangles** rendered from the Moana Island Scene

---

## Future Work

- **Two-level BVH** for proper instancing (needed for full Moana with 30M instances)
- **Volumetric rendering** (fog, smoke, participating media)
- **Bidirectional path tracing** for difficult light transport (caustics through glass)
- **GPU denoising** (OIDN CUDA backend or OptiX denoiser)
- **Motion blur** and time-sampled ray generation
- **IES light profiles** for architectural lighting

---

*Photon is built with C++23, Kokkos 5.0, ANARI SDK 0.15.0, and optionally Embree 4.3.3, OptiX 8.1, and Intel OIDN 2.2.*
