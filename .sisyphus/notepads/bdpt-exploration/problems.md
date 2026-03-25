# BDPT Implementation - Problems & Gotchas

## Problem 1: Light Point Sampling Gap
**Status**: BLOCKING for light path generation
**Severity**: HIGH
**Description**:
- Current `sample_light()` computes solid angle PDF at hit_pos
- To start light paths, need: "sample a point on light source"
- This requires different PDF (area, not solid angle)
- Current implementation doesn't provide this

**Impact**:
- Cannot trace rays from light sources
- Needed for light path generation phase

**Solution**:
- Implement `sample_light_point()` in light.h
- Handles all light types differently:
  - Point light: single point, pdf = 1
  - Area light: uniform sample on quad, pdf = 1 / area
  - Directional: need to handle at infinity (see below)

**Directional Light Issue**:
- Directional lights have no position/area
- Standard BDPT doesn't handle parallel light sources well
- Options:
  1. Skip directional lights in light path generation
  2. Use special-case handling (project to near plane)
  3. Use Russian roulette to limit contribution
  - Recommend: Option 1 for now, handle as pure NEE

## Problem 2: Camera Reverse Function Missing
**Status**: BLOCKING for connection kernel
**Severity**: HIGH
**Description**:
- Need: given world point, compute PDF of sampling it from camera
- Requires inverse of perspective projection
- Must handle DOF (thin lens model)

**Math required**:
```
Given: P_world
1. Transform to camera space: P_cam = inv(camera_matrix) * P_world
2. Project to image plane: (x_film, y_film)
3. Compute solid angle PDF:
   pdf_solid = pdf_area / (dist^2 * |cos_theta|)
   where dist = distance from camera lens to point
         cos_theta = dot(direction, forward)
4. Handle DOF: coordinate with lens distribution
```

**Complexity**:
- Thin lens model has disk of possible origins on lens
- Must integrate PDF over all lens positions
- Or simplify: assume pinhole camera (lens_radius = 0)

**Solution approach**:
- Start with pinhole camera (no DOF)
- Implement projection logic
- Store PDF in CameraReverseResult
- Can add DOF later if needed

## Problem 3: Path Vertex Storage Architecture
**Status**: BLOCKING for full BDPT
**Severity**: HIGH
**Description**:
- Current code tracks only: position, normal, material, throughput, pdf
- Need to also track:
  - pdf_fwd: PDF of sampling in sampled direction
  - pdf_bwd: PDF of reverse direction (for MIS)
  - is_delta: whether bounce was specular
  - depth: distance from start (for MIS combinations)

**Storage challenge**:
- Kokkos views require pre-allocation
- Don't know how many vertices per path ahead of time
- Wavefront PT: different rays at different depths

**Solution**:
- Pre-allocate view: [pixel_count] x [max_depth]
- Store: PathVertex array for each pixel/depth pair
- Store validity flag for each vertex
- Wastes some memory (unfilled slots) but enables parallel access

## Problem 4: Visibility Testing Between Arbitrary Points
**Status**: BLOCKING for connection kernel
**Severity**: HIGH
**Description**:
- NEE uses shadow rays: camera hit -> light
- BDPT needs: camera vertex i -> light vertex j
- Generalizes but needs unidirectional ray tracing

**Current API**:
```cpp
void trace_occluded(const RayBatch &rays, 
                   Kokkos::View<u32 *> occluded);
```
- Already supports batch ray tracing!
- Just need to generate rays between arbitrary points

**Solution**:
- In connection kernel: for each camera/light vertex pair
- Construct ray from camera vertex toward light vertex
- Clamp ray tmax to distance (0.99 * dist to avoid self-intersection)
- Reuse trace_occluded infrastructure

**Subtlety**: Self-intersection avoidance
- Both endpoints are at surface
- Need epsilon offset in ray tmin
- Code already does this (pathtracer.cpp line 279)

## Problem 5: MIS Weight Computation Complexity
**Status**: BLOCKING for correct MIS
**Severity**: HIGH
**Description**:
- Current MIS: compares 2 strategies (light sample vs BSDF sample)
- BDPT MIS: must compare ALL connection strategies of same length
- Example for length 3 paths: 
  - (s=1, t=2): light 1 bounce, camera 2 bounces
  - (s=2, t=1): light 2 bounces, camera 1 bounce

**Math complexity**:
```
weight = 1 / (sum over all (s',t') of (pdf_s' * pdf_t')^2)
```
- For each vertex, need pdf of arriving AND departing
- Must store both directions at each vertex
- Multiple path contributions share same geometry

**Algorithm**:
1. Store pdf_fwd (arriving) and pdf_bwd (departing) at each vertex
2. For connection s,t:
   - Compute product of all forward/backward PDFs along both paths
   - Iterate over all (s',t') with s'+t' = length
   - Sum their PDF products
   - Apply power heuristic

**Performance gotcha**:
- MIS computation is O(max_depth^2) per connection
- Repeated per pixel
- May need optimization (precomputation, caching)

## Problem 6: Specular Path Handling
**Status**: DESIGN DECISION NEEDED
**Severity**: MEDIUM
**Description**:
- Specular bounces have delta distribution (PDF = 0 elsewhere)
- Cannot connect to arbitrary directions
- Example: perfect mirror can only connect if aligned

**Current support**:
- BsdfSample.is_specular flag exists
- Current code checks: `bs.is_specular` (pathtracer.cpp line 387)

**Challenge for BDPT**:
- If path has delta bounce, what's the MIS weight?
- PDF of specular path = 0 for non-specular direction
- Still valid to connect if both paths are specular

**Solution approach**:
- Store is_delta at each vertex
- In connection kernel:
  - If both endpoints preceded by delta: skip connection
  - If one endpoint preceded by delta: weight = 0 contribution
  - Full MIS computation for non-delta paths
- OR: simplified: never connect after delta bounces (conservative)

## Problem 7: Emissive Surface Light Path Generation
**Status**: NICE TO HAVE initially
**Severity**: MEDIUM
**Description**:
- Scene has emissive_prim_ids and emissive_prim_areas
- These aren't in Light[] array
- Need separate sampling logic

**Current data**:
```cpp
Kokkos::View<u32 *> emissive_prim_ids;
Kokkos::View<f32 *> emissive_prim_areas;
f32 total_emissive_area;
u32 emissive_count;
```

**Challenge**:
- Must sample triangle from mesh
- Need positions to form ray origin
- Need normal for emission direction PDF

**Solution**:
- Extend sample_light_point() to handle emissive flag
- OR: Unify emissive sampling with Light sampling
- Triangle sampling: barycentric coords (see pathtracer.cpp lines 294-299)

**Note**: Can defer to Phase 2, focus on analytic lights first

## Problem 8: Environment Map Light Paths
**Status**: CAN DEFER
**Severity**: LOW initially
**Description**:
- EnvironmentMap has sample_direction()
- Infinite distance: special PDF handling
- Creates rays in different sampling pattern

**Challenge**:
- Directional distribution vs area distribution
- PDF at infinity involves solid angle only
- Connection must handle infinite distance

**Solution**:
- Defer to Phase 2
- Focus on analytic + emissive lights first
- Can add env map contribution as pure NEE (existing)

## Problem 9: Wavefront PT Memory Layout
**Status**: DESIGN DECISION
**Severity**: MEDIUM
**Description**:
- Wavefront PT: different rays at different depths
- Active pixel count changes each bounce
- Path vertices for dead rays are wasted memory

**Current approach**:
- Views pre-allocated to pixel_count
- Inactive rays set tmax = 0 (pathtracer.cpp line 420)
- Shade kernel checks active() flag (line 166)

**For BDPT**:
- Same issue: need views for all depths
- Pre-allocate [pixel_count] x [max_depth] x [vertex_data]
- Mark invalid vertices with is_valid flag
- Parallel kernels skip invalid vertices

**Memory tradeoff**:
- Simple: allocate full size, accept waste
- Optimized: compact storage, increases kernel complexity
- Recommend: start simple, optimize if needed

## Problem 10: Debugging and Validation
**Status**: OPERATIONAL
**Severity**: MEDIUM
**Description**:
- BDPT is complex: many moving parts
- Easy to introduce subtle bugs
- Incorrect MIS weighting hard to spot

**Validation strategy**:
1. **Phase 1**: Render with light paths only (no connections)
   - Should be darker than full render (missing camera direct)
2. **Phase 2**: Single connection per camera vertex
   - Render with BDPT depth 1
3. **Phase 3**: Progressive depth increases
   - Compare against unidirectional (should match or beat)
4. **Phase 4**: MIS verification
   - Render with/without MIS (should be similar variance)
5. **Phase 5**: Reference comparison
   - Use known scene + reference renderer

**Debugging tools**:
- Visualize light path density
- Output MIS weights for analysis
- Compare pixel by pixel differences
- Statistical variance analysis

## Summary of Blocking Issues

| Issue | File | Scope |
|-------|------|-------|
| Light point sampling | light.h | Add 2 functions |
| Camera reverse | camera.h | Add 1 function |
| Path vertex storage | (new file?) | New struct + management |
| Connection kernel | (new code) | Major new logic |
| MIS computation | (new code) | Complex algorithm |
| Specular handling | (extensions) | Flag + logic |
| Emissive sampling | light.h | Extend existing |
| Memory layout | pathtracer.cpp | Moderate change |
