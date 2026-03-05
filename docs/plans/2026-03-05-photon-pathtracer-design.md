# Photon Path Tracer — Full Design

**Date**: 2026-03-05
**Status**: Approved
**Scope**: Complete the ANARI+Kokkos path tracer with modern features, hardware-accelerated ray-tracing backends, and full PBR materials.

---

## 1. Architecture Overview

Photon uses a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────┐
│              ANARI Device Layer                  │
│    PhotonDevice, SceneFromAnari, ANARI objects   │
├─────────────────────────────────────────────────┤
│         Wavefront Path Tracing Integrator        │
│  (Kokkos parallel_for — portable across all HW) │
│  NEE, MIS, Russian Roulette, Disney BSDF, SSS   │
├─────────────────────────────────────────────────┤
│             RayBackend Interface                 │
│        build_accel() / trace_rays()              │
├──────────┬──────────┬──────────┬────────────────┤
│  Kokkos  │  Embree  │  OptiX   │   HIP RT       │
│   BVH    │  (CPU)   │ (NVIDIA) │   (AMD)        │
│(fallback)│ AVX-512  │ RT cores │   RDNA2+       │
└──────────┴──────────┴──────────┴────────────────┘
```

### Design Decisions

1. **Wavefront integrator**: The path tracer generates ray batches per bounce level. The RayBackend traces them. Shading happens in Kokkos. This naturally maps to both CPU (Embree batch) and GPU (OptiX/HIPRT kernel launch).

2. **Kokkos for shading**: All material evaluation, light sampling, and path logic stays in Kokkos (`KOKKOS_FUNCTION`). This keeps shading portable across CPU, CUDA, HIP, and SYCL without rewriting per-backend.

3. **Backends for intersection only**: Embree/OptiX/HIPRT handle BVH build and ray-primitive intersection. They return hit results (t, normal, prim_id, UV). The integrator does everything else.

4. **Auto-detection**: At startup, Photon probes available hardware and selects the fastest backend. User can override via environment variable `PHOTON_BACKEND=embree|optix|hiprt|kokkos`.

---

## 2. Ray Backend Interface

```cpp
namespace photon::pt {

struct HitResult {
  f32 t;
  Vec3 normal;
  Vec3 position;
  f32 u, v;            // barycentric coordinates
  u32 prim_id;
  u32 geom_id;         // which geometry in scene
  u32 inst_id;         // which instance
  bool hit;
};

// Batch of rays for wavefront tracing
struct RayBatch {
  Kokkos::View<Vec3*> origins;
  Kokkos::View<Vec3*> directions;
  Kokkos::View<f32*>  tmin;
  Kokkos::View<f32*>  tmax;
  u32 count;
};

struct HitBatch {
  Kokkos::View<HitResult*> hits;
  u32 count;
};

struct RayBackend {
  virtual ~RayBackend() = default;

  // Build acceleration structure from scene geometry
  virtual void build_accel(const Scene& scene) = 0;

  // Trace closest-hit rays (returns hit info)
  virtual void trace_closest(const RayBatch& rays, HitBatch& hits) = 0;

  // Trace shadow/occlusion rays (returns bool per ray)
  virtual void trace_occluded(const RayBatch& rays,
                              Kokkos::View<u32*> occluded) = 0;

  // Name for logging
  virtual const char* name() const = 0;
};

// Factory: auto-detect best backend
std::unique_ptr<RayBackend> create_best_backend();

} // namespace photon::pt
```

### Backend Implementations

| Backend | When Selected | BVH Build | Intersection | Notes |
|---------|---------------|-----------|-------------|-------|
| `KokkosBackend` | Always available (fallback) | CPU median-split (existing) | `intersect_mesh_bvh()` via Kokkos | Portable everywhere |
| `EmbreeBackend` | Embree 4 linked | `rtcBuildBVH` (SAH) | `rtcIntersect1/4/8/16` | CPU SIMD, best for Intel |
| `OptixBackend` | NVIDIA GPU detected | OptiX AS build | `optixTrace` | RT core acceleration |
| `HiprtBackend` | AMD GPU detected | `hiprtBuildGeometry` | `hiprtTrace` | RDNA2+ ray accelerators |

### Auto-Detection Logic

```
1. Check if PHOTON_BACKEND env var is set → use that
2. If Kokkos::Cuda is active → try OptiX
3. If Kokkos::HIP is active → try HIP RT
4. If Embree is linked → use Embree (best CPU)
5. Fallback → Kokkos BVH
```

---

## 3. Material System — Disney Principled BSDF + SSS

### Material Structure

```cpp
struct Material {
  // Disney Principled parameters
  Vec3 base_color{0.8f, 0.8f, 0.8f};
  f32  metallic{0.f};
  f32  roughness{0.5f};
  f32  ior{1.5f};
  f32  transmission{0.f};
  f32  specular{0.5f};
  f32  clearcoat{0.f};
  f32  clearcoat_roughness{0.03f};

  // Emission
  Vec3 emission{0.f, 0.f, 0.f};
  f32  emission_strength{0.f};

  // Subsurface scattering
  f32  subsurface{0.f};
  Vec3 subsurface_color{1.f, 1.f, 1.f};
  f32  subsurface_radius{1.f};

  // Texture IDs (-1 = no texture)
  i32 base_color_tex{-1};
  i32 normal_tex{-1};
  i32 roughness_tex{-1};
  i32 metallic_tex{-1};
  i32 emission_tex{-1};
};
```

### BSDF Evaluation

The Disney BSDF has three lobes evaluated per hit:

1. **Diffuse lobe** — Lambertian diffuse weighted by `(1 - metallic) * (1 - transmission)`. Uses cosine-weighted hemisphere sampling.

2. **Specular reflection lobe** — GGX microfacet BRDF. Uses visible normal distribution (VNDF) importance sampling. Fresnel is Schlick approximation for dielectrics, complex Fresnel for metals (tinted by base_color).

3. **Transmission lobe** — GGX microfacet BTDF for refraction, weighted by `transmission * (1 - metallic)`. Snell's law with total internal reflection.

4. **Clearcoat lobe** — Second GGX specular layer with fixed IOR=1.5, weighted by `clearcoat`.

5. **Subsurface scattering** — Random-walk SSS below the surface when `subsurface > 0`. Replaces diffuse component. Uses exponential free-flight sampling with `subsurface_radius` as mean free path.

**Lobe selection**: At each bounce, randomly pick a lobe proportional to its weight. Evaluate the selected lobe's BSDF value and PDF, apply MIS.

### Texture System

```cpp
struct Texture {
  Kokkos::View<Vec3**, Kokkos::LayoutRight> pixels; // width x height
  u32 width, height;
  // Bilinear sampling
  KOKKOS_FUNCTION Vec3 sample(f32 u, f32 v) const;
};

struct TextureAtlas {
  Kokkos::View<Texture*> textures;
  u32 count;
};
```

Textures stored as Kokkos Views so they're accessible on device. Bilinear interpolation in the `sample()` function.

---

## 4. Lighting

### Light Types

```cpp
enum class LightType : u32 {
  Point,
  Directional,
  Spot,
  Area,       // emissive triangle mesh
  Environment // HDR environment map
};

struct Light {
  LightType type;
  Vec3 position;         // point/spot
  Vec3 direction;        // directional/spot
  Vec3 color;
  f32  intensity;
  f32  spot_angle;       // spot cone half-angle
  f32  spot_falloff;     // spot edge softness
  u32  mesh_prim_begin;  // area light: first emissive primitive
  u32  mesh_prim_count;  // area light: number of emissive primitives
};
```

### Environment Map

```cpp
struct EnvironmentMap {
  Kokkos::View<Vec3**, Kokkos::LayoutRight> pixels; // equirectangular HDR
  Kokkos::View<f32*>  cdf;        // marginal CDF for importance sampling
  Kokkos::View<f32**> cond_cdf;   // conditional CDF per row
  u32 width, height;

  KOKKOS_FUNCTION Vec3 sample_direction(Rng& rng, f32& pdf) const;
  KOKKOS_FUNCTION Vec3 evaluate(const Vec3& direction) const;
  KOKKOS_FUNCTION f32  pdf(const Vec3& direction) const;
};
```

The environment map uses hierarchical 2D CDF for importance sampling (standard technique from PBRT). This gives low-variance samples toward bright regions of the HDR map.

### Area Lights

Any mesh primitive with `emission_strength > 0` is automatically an area light. Light sampling picks a random emissive triangle proportional to its area, then samples a point on it.

---

## 5. Sampling & Integration

### Wavefront Path Tracing Loop

```
for each sample:
  1. Generate camera rays (batch)             ← Kokkos parallel_for
  2. for each bounce (up to max_depth):
     a. trace_closest(rays) → hits            ← RayBackend
     b. Shade hits:                           ← Kokkos parallel_for
        - Evaluate material at hit point
        - NEE: sample light, trace shadow ray
        - BSDF: sample new direction
        - MIS: combine light + BSDF weights
        - Russian roulette: probabilistic termination
        - Generate new rays for next bounce
     c. trace_occluded(shadow_rays) → visible  ← RayBackend
     d. Add direct light contribution where visible
  3. Accumulate radiance to pixel buffer
```

### Next Event Estimation (NEE)

At each non-specular hit, explicitly sample a light source:
1. Pick a light (uniform or power-weighted)
2. Sample a point on the light
3. Compute shadow ray from hit point to light sample
4. If unoccluded: add `Le * BSDF * cos_theta / (pdf_light * pdf_light_select)`
5. Apply MIS weight: `w = power_heuristic(pdf_light, pdf_bsdf)`

### Multiple Importance Sampling (MIS)

Power heuristic with β=2:
```cpp
KOKKOS_FUNCTION f32 power_heuristic(f32 pdf_f, f32 pdf_g) {
  f32 f2 = pdf_f * pdf_f;
  return f2 / (f2 + pdf_g * pdf_g);
}
```

Applied between:
- Light sampling PDF and BSDF sampling PDF (for direct illumination)
- BSDF sampling and environment map PDF (for environment lighting)

### Russian Roulette

After bounce 3, termination probability based on max throughput component:
```cpp
f32 continue_prob = min(max_component(throughput), 0.95f);
if (rng.next_f32() > continue_prob) break;
throughput /= continue_prob;
```

### Sampler

Replace xorshift with **Sobol sequence** (Owen-scrambled) for better convergence:
```cpp
struct SobolSampler {
  u32 pixel_index;
  u32 sample_index;
  u32 dimension;

  KOKKOS_FUNCTION f32 next_1d();
  KOKKOS_FUNCTION Vec2 next_2d();
};
```

Sobol direction numbers stored in a small constant Kokkos::View. Falls back to PCG RNG for dimensions beyond the Sobol table.

---

## 6. Volumetric Rendering

### Volume Data

```cpp
struct VolumeGrid {
  Kokkos::View<f32***, Kokkos::LayoutRight> density;  // NxNxN
  Vec3 bounds_lo, bounds_hi;
  f32 max_density;  // majorant for delta tracking

  Vec3 sigma_s;     // scattering coefficient (color)
  Vec3 sigma_a;     // absorption coefficient
  f32  g;           // Henyey-Greenstein asymmetry parameter
  Vec3 emission;    // volume emission color
  f32  emission_strength;
};
```

### Delta Tracking (Woodcock Tracking)

Unbiased free-flight sampling through heterogeneous media:
```
1. At ray entry into volume bbox:
2. Sample free-flight distance: t = -log(rng) / (sigma_t_majorant)
3. If t exits volume: no interaction (transmitted)
4. Sample density at position: rho = density_grid.sample(position)
5. If rng < rho / max_density: real interaction (scatter or absorb)
6. Else: null interaction (continue sampling from current position)
```

### Phase Function

Henyey-Greenstein for anisotropic scattering:
```cpp
KOKKOS_FUNCTION Vec3 sample_hg(const Vec3& wo, f32 g, Rng& rng);
KOKKOS_FUNCTION f32  pdf_hg(f32 cos_theta, f32 g);
```

### Integration with Path Tracer

When a ray enters a volume, delta tracking decides if scattering occurs before the next surface hit. If scattering: apply phase function, sample new direction, continue path. The integrator handles surface-volume transitions naturally.

---

## 7. Scene Structure (Expanded)

```cpp
struct Scene {
  // Geometry
  Kokkos::View<TriangleMesh*> meshes;
  Kokkos::View<u32*>          mesh_material_ids;  // material per mesh
  u32 mesh_count;

  // Materials
  Kokkos::View<Material*> materials;
  u32 material_count;

  // Textures
  TextureAtlas textures;

  // Lights
  Kokkos::View<Light*> lights;
  u32 light_count;
  std::optional<EnvironmentMap> env_map;

  // Volumes
  Kokkos::View<VolumeGrid*> volumes;
  u32 volume_count;

  // Instances (transform + mesh_id)
  Kokkos::View<Instance*> instances;
  u32 instance_count;
};

struct Instance {
  Mat4 transform;
  Mat4 inv_transform;
  u32 mesh_id;
  u32 material_id;
};
```

---

## 8. ANARI Device Completion

### Objects to Implement

| ANARI Object | Status | Implementation |
|-------------|--------|---------------|
| Geometry (triangle, quad, sphere) | Existing + extend | Add sphere, smooth normals, UVs |
| Surface | Existing | Add material reference |
| Material (matte, physicallyBased) | **New** | Map to Disney BSDF parameters |
| Camera (perspective, orthographic) | **New** | Thin lens DOF, ANARI params |
| Light (point, directional, hdri) | **New** | Map to Light struct |
| Volume (transferFunction1D) | **New** | Map to VolumeGrid |
| SpatialField (structuredRegular) | **New** | Grid-based density |
| Group | **New** | Geometry collection |
| Instance | **New** | Transform + group reference |
| World | Existing | Add lights, volumes, instances |
| Frame | Existing | Add AOV channels, progressive mode |
| Renderer | Existing | Add SPP, max_depth, background params |

### SceneFromAnari Refactor

Extract the 200+ line inline scene builder from `PhotonDevice::renderFrame()` into `build_scene_from_anari()`. This function walks the ANARI object graph and builds the full `Scene` struct including materials, lights, textures, and instances.

---

## 9. Post-Processing

### Tone Mapping

ACES filmic tone mapping applied in `frameBufferMap()`:
```cpp
KOKKOS_FUNCTION Vec3 aces_tonemap(Vec3 color) {
  const f32 a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  return clamp01((color * (a * color + b)) / (color * (c * color + d) + e));
}
```

### AOVs (Arbitrary Output Variables)

Frame channels beyond "color":
- `"color"` — beauty pass (FLOAT32_VEC4)
- `"depth"` — Z distance (FLOAT32)
- `"normal"` — shading normal (FLOAT32_VEC3)
- `"albedo"` — surface albedo (FLOAT32_VEC3)

AOV buffers allocated alongside the main pixel buffer. Normal and albedo channels are needed by OIDN denoiser.

### Denoising (Intel OIDN)

Optional post-process using Intel Open Image Denoise:
- Input: noisy color + albedo AOV + normal AOV
- Output: denoised color
- Works on both CPU and GPU
- Dramatically improves quality at low sample counts (4-16 SPP)
- Enabled via renderer parameter `"denoise"` = true

---

## 10. Progressive Rendering

### Accumulation Buffer

```cpp
struct FrameState {
  Kokkos::View<Vec3**, Kokkos::LayoutRight> accum;  // accumulated radiance
  u32 samples_completed;
  u32 samples_requested;
  bool is_complete;
};
```

### API Flow

- `renderFrame()` kicks off rendering (potentially async)
- `frameReady(ANARI_NO_WAIT)` returns 0 if still rendering, 1 if done
- `frameReady(ANARI_WAIT)` blocks until complete
- `frameBufferMap("color")` returns current accumulated result (even if incomplete)
- Each call to `renderFrame()` on a non-changed scene adds more samples

### Batch Mode

When `samples_per_pixel` is set on the renderer, all samples render in one `renderFrame()` call. This is the HPC/batch workflow.

---

## 11. Build System Changes

### New CMake Options

```cmake
option(PHOTON_ENABLE_EMBREE  "Enable Embree ray tracing backend"  ON)
option(PHOTON_ENABLE_OPTIX   "Enable OptiX ray tracing backend"   OFF)
option(PHOTON_ENABLE_HIPRT   "Enable HIP RT ray tracing backend"  OFF)
option(PHOTON_ENABLE_OIDN    "Enable Intel OIDN denoiser"         ON)
```

Embree and OIDN default ON because they work on CPU. OptiX and HIPRT default OFF because they require vendor SDKs.

### Dependencies

| Dependency | Version | Method | Notes |
|-----------|---------|--------|-------|
| Kokkos | 5.0.1 | FetchContent | Existing |
| ANARI SDK | 0.15.0 | FetchContent | Existing |
| Embree | 4.x | find_package or FetchContent | CPU ray tracing |
| OptiX | 7.x+ | find_package (headers only) | NVIDIA RT cores |
| HIP RT | ROCm 5.x+ | find_package | AMD ray tracing |
| OIDN | 2.x | find_package or FetchContent | Denoiser |
| stb_image | latest | FetchContent | HDR/PNG/JPEG loading |

---

## 12. File Organization

```
photon/
├── include/photon/
│   ├── pt/
│   │   ├── math.h, math_aabb.h, ray.h     (existing)
│   │   ├── camera.h                         (extend: thin lens)
│   │   ├── material.h                       (rewrite: Disney BSDF)
│   │   ├── texture.h                        (new)
│   │   ├── light.h                          (new)
│   │   ├── environment_map.h                (new)
│   │   ├── volume.h                         (new)
│   │   ├── scene.h                          (extend: full scene)
│   │   ├── pathtracer.h                     (rewrite: wavefront)
│   │   ├── sampling.h                       (new: MIS, Sobol, etc.)
│   │   ├── disney_bsdf.h                    (new)
│   │   ├── tone_mapping.h                   (new)
│   │   ├── geom/
│   │   │   ├── triangle_mesh.h              (extend: normals, UVs)
│   │   │   ├── triangle_intersect.h         (existing)
│   │   │   ├── mesh_intersector.h           (existing)
│   │   │   └── sphere.h                     (move from pt/sphere.h)
│   │   ├── bvh/
│   │   │   ├── bvh.h                        (existing)
│   │   │   └── bvh_validate.h               (existing)
│   │   └── backend/
│   │       ├── ray_backend.h                (new: interface)
│   │       ├── kokkos_backend.h             (new: fallback)
│   │       ├── embree_backend.h             (new)
│   │       ├── optix_backend.h              (new)
│   │       └── hiprt_backend.h              (new)
│   └── anari/
│       ├── device.h, library.h              (existing)
│       └── (internal headers stay in src/)
├── src/photon/
│   ├── pt/
│   │   ├── pathtracer.cpp                   (rewrite: wavefront)
│   │   ├── disney_bsdf.cpp                  (new)
│   │   ├── environment_map.cpp              (new)
│   │   ├── volume.cpp                       (new)
│   │   ├── scene.cpp                        (extend)
│   │   ├── backend/
│   │   │   ├── kokkos_backend.cpp           (new)
│   │   │   ├── embree_backend.cpp           (new)
│   │   │   ├── optix_backend.cpp            (new)
│   │   │   ├── hiprt_backend.cpp            (new)
│   │   │   └── backend_factory.cpp          (new: auto-detect)
│   │   └── ...
│   └── anari/
│       ├── PhotonDevice.cpp                 (refactor renderFrame)
│       ├── SceneFromAnari.cpp               (implement fully)
│       └── ...
└── docs/plans/
    └── 2026-03-05-photon-pathtracer-design.md
```
