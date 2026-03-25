# BDPT Codebase Analysis - Key Learnings

## Codebase Architecture
The Photon path tracer is a **wavefront path tracer** using Kokkos for GPU parallelization.

### Main Loop Structure (pathtracer.cpp)
- **Samples loop**: iterates over samples_per_pixel
- **Bounce loop**: iterates up to max_depth
- **Per-bounce phases**:
  1. Trace rays to closest hit (trace_closest)
  2. Shade: compute NEE, evaluate BSDF, sample next bounce
  3. Trace shadow rays (trace_occluded)
  4. Accumulate NEE contributions

### Critical Path Through Code
1. Initialize camera ray: `cam.ray(u, v, rng)` → Ray
2. Trace: `m_backend->trace_closest(rays, hits)` → HitResult[]
3. Shade kernel: processes each hit in parallel
   - Material lookup and texture sampling
   - Emission detection
   - NEE with `sample_light()` + shadow rays
   - BSDF sampling with `disney_bsdf_sample()`
4. Shadow test: `m_backend->trace_occluded(shadow_rays, occluded)`
5. Accumulate visible NEE contributions

## What Already Works for BDPT

### BSDF Infrastructure (disney_bsdf.h)
- **BsdfSample struct** (line 12): has wi, f, pdf, **is_specular flag**
- `disney_bsdf_sample()`: returns full BsdfSample
- `disney_bsdf_eval()`: evaluates BSDF (needed for reverse path MIS)
- `disney_bsdf_pdf()`: evaluates PDF (needed for reverse path MIS)
- ✅ **All three functions can be called with any (wo, wi) pair** - perfect for bidirectional evaluation

### Light Sampling (light.h)
- `sample_light()` at lines 53-115
- Returns LightSample with wi, Li, pdf, dist, is_delta
- Handles: Point, Directional, Spot, Area lights
- ✅ **Supports both delta and non-delta distributions**

### Ray Backend (ray_backend.h)
- trace_closest: batch ray tracing
- trace_occluded: batch visibility testing
- ✅ **Can trace between any two points**

### Scene Data (scene.h)
- mesh (positions, normals, indices)
- materials[]
- lights[]
- emissive_prim_ids[], emissive_prim_areas[]
- env_map (optional)
- ✅ **All needed for light path generation**

## Critical Gaps for BDPT

### 1. Light Point Sampling
- `sample_light()` returns PDF at hit_pos (solid angle from surface)
- **Missing**: function to sample point on light source + return PDF of point
- **Needed for**: light subpath generation
- **Architecture impact**: light paths need explicit vertex storage

### 2. Camera Reverse Function
- `Camera::ray()` generates ray from (s,t)
- **Missing**: inverse function to get (s,t) from world point
- **Needed for**: evaluating camera subpath PDF at connection point
- **Implementation**: project world point to camera space, get screen coords, compute PDF

### 3. Path Vertex Storage
- Current code uses implicit representation: throughput only
- **Missing**: explicit vertex structure for bidirectional paths
- **Needed for**:
  - Storing camera path vertices
  - Storing light path vertices
  - Connecting arbitrary pairs of vertices
- **Design**: PathVertex = {position, normal, material, is_delta_bounce, pdf_fwd, pdf_bwd}

### 4. Connection Kernel
- Current NEE: one-way camera-to-light
- **Missing**: kernel to connect camera vertex to light vertex
- **Challenges**:
  - Visibility testing between arbitrary points
  - MIS weight computation for all s,t combinations
  - Specular path handling

## MIS Framework Already Present
- `power_heuristic()` used at line 270: `power_heuristic(1, ls.pdf, 1, bsdf_pdf)`
- Pattern: compares two sampling strategies
- **Can extend** for BDPT with full power heuristic over all connection strategies

## Storage/Parallelization Notes
- All data in Kokkos views (GPU-friendly)
- Parallel kernels use `Kokkos::parallel_for`
- Per-pixel RNG seed from: `Rng rng(u32(1337u) ^ (u32(x) * 9781u) ^ (u32(y) * 6271u) ^ (s * 26699u))`
- Adaptive sampling support already present (lines 99-124)

## Key Function Signatures for Reference

### BSDF
```cpp
Vec3 disney_bsdf_eval(const Material &mat, const Vec3 &wo, const Vec3 &wi, 
                      const Vec3 &n, const Vec3 &shading_n);  // Line 201
f32 disney_bsdf_pdf(const Material &mat, const Vec3 &wo, const Vec3 &wi, 
                    const Vec3 &n, const Vec3 &shading_n);    // Line 221
BsdfSample disney_bsdf_sample(const Material &mat, const Vec3 &wo, 
                              const Vec3 &n, const Vec3 &shading_n, Rng &rng); // Line 293
```

### Light
```cpp
LightSample sample_light(const Light &light, const Vec3 &hit_pos, Rng &rng); // Line 53
f32 light_pdf(const Light &light, const Vec3 &hit_pos, const Vec3 &wi);      // Line 117
```

### Ray Backend
```cpp
void trace_closest(const RayBatch &rays, HitBatch &hits);
void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded);
```

### Camera
```cpp
Ray ray(f32 s, f32 t, Rng &rng) const; // Line 80
```

## Implementation Hints

### Light Path Generation
- Similar to camera path loop, but start from light
- `sample_light_point()` → position on light + emission + normal
- Trace ray in light's outgoing direction
- Build vertex chain storing position, normal, material, pdf

### Camera Reverse
- Project camera-space point to screen
- `(s, t) = ((x_cam + width/2) / width, (y_cam + height/2) / height)`
- PDF includes solid angle + Jacobian of projection

### Connection
- For camera vertex C and light vertex L:
  - Direct distance vector
  - Shadow trace to test visibility
  - Evaluate BSDF eval at both vertices
  - Evaluate PDFs in reverse directions
  - Compute MIS weight

### MIS Weight for (s,t) connection
- Multiple strategies: s-segment light path + t-segment camera path
- Full power heuristic over all possible (s',t') with same geometry
- Simplification: can compute using pdf_fwd and pdf_bwd stored at vertices
