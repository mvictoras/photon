# BDPT Implementation Architecture Decisions

## Decision 1: Light Path Representation
**Decision**: Store light paths as explicit vertex chains in Kokkos views
**Rationale**:
- Current code uses implicit paths (throughput only)
- BDPT requires evaluating PDFs in reverse directions
- Need to connect arbitrary pairs of vertices
- GPU parallelization benefits from explicit data structures

**Structure design**:
```cpp
struct PathVertex {
  Vec3 position;
  Vec3 normal;
  Vec3 shading_normal;
  u32 material_id;
  Vec3 throughput;        // Accumulated contribution
  f32 pdf_fwd;           // PDF of arriving at this vertex
  f32 pdf_bwd;           // PDF of leaving this vertex in reverse
  bool is_delta;         // Was this a delta bounce?
};
```

## Decision 2: Light Sampling Functions
**Decision**: Add `sample_light_point()` to light.h
**Rationale**:
- Current `sample_light()` returns PDF at hit_pos (solid angle)
- For light paths, need to sample a point on light + return its area PDF
- Needed before ray tracing from light

**New functions**:
```cpp
struct LightPointSample {
  Vec3 position;
  Vec3 normal;
  Vec3 L;              // Emission
  f32 pdf;             // Area PDF (not solid angle)
};

KOKKOS_FUNCTION inline LightPointSample sample_light_point(
  const Light &light, Rng &rng);

KOKKOS_FUNCTION inline f32 light_point_pdf(
  const Light &light, const Vec3 &point);
```

## Decision 3: Camera Reverse Function
**Decision**: Add `camera_reverse()` to camera.h
**Rationale**:
- Current `ray()` generates ray from (s,t)
- For BDPT, need to evaluate PDF of sampling a world point from camera
- Must account for perspective projection + DOF

**New function**:
```cpp
struct CameraReverseResult {
  f32 screen_x, screen_y;
  f32 pdf;              // PDF in solid angle
};

KOKKOS_FUNCTION inline CameraReverseResult camera_reverse(
  const Vec3 &world_point);
```

## Decision 4: BDPT Integration Point
**Decision**: Extend PathTracer to generate both camera and light paths
**Rationale**:
- Minimal disruption to existing code
- Reuse existing parallel infrastructure
- Can phase in BDPT connections gradually

**Structure**:
1. Keep existing camera path generation
2. Add parallel light path generation kernel
3. Add connection kernel between path vertices
4. Enhance shading kernel to handle both old NEE and new BDPT connections

## Decision 5: MIS Weight Computation
**Decision**: Implement full power heuristic for all (s,t) combinations
**Rationale**:
- Current power_heuristic works for 2-strategy case
- BDPT requires weighting over multiple connection strategies
- Can cache pdf_fwd and pdf_bwd at each vertex

**Algorithm**:
```
For connection at depth s (light) and depth t (camera):
  1. Compute pdf of this path via BSDF forward
  2. Compute pdf of reverse direction via BSDF backward
  3. For each s',t' where s'+t' = s+t:
     - Compute pdf of that strategy
  4. Apply power heuristic
```

## Decision 6: Specular Path Handling
**Decision**: Terminate specular paths or mark for connection validity
**Rationale**:
- is_specular flag already in BsdfSample
- Specular bounces have zero PDF for non-specular directions
- Must avoid invalid connections

**Approach**:
- Store is_delta flag in PathVertex
- In connection kernel: skip if either endpoint is preceded by delta
- Or: use appropriate PDF handling (zero for impossible connections)

## Decision 7: Emissive Surface Support
**Decision**: Handle emissives similarly to analytic lights
**Rationale**:
- Scene already tracks emissive_prim_ids and emissive_prim_areas
- Can sample light ray origins from emissive triangles
- Same connection logic as analytic lights

**Implementation**:
- Extend light path generation to include emissive surfaces
- Unify sampling logic in LightPointSample

## Decision 8: Environment Map Integration
**Decision**: Treat as special case of area light
**Rationale**:
- EnvironmentMap has sample_direction() already
- BDPT connections to env still valid
- Special PDF handling for infinite distance

**Note**: May defer full env map light paths to Phase 2

## Decision 9: Adaptive Sampling with BDPT
**Decision**: Keep adaptive sampling framework, extend to BDPT
**Rationale**:
- Code already has variance tracking per pixel
- BDPT reduces variance, so adaptive sampling still valuable
- Can apply same per-pixel logic

**Change**:
- Variance computation already sums contributions
- BDPT just changes where contributions come from
- No structural change needed

## Decision 10: Testing Strategy
**Decision**: Implement in phases with comparison to current path tracer
**Rationale**:
- Phase 1: Light path generation (verify with debug output)
- Phase 2: Camera reverse function (verify coordinates)
- Phase 3: Connections (compare per-light-depth contributions)
- Phase 4: Full BDPT (compare final images)
- Phase 5: Optimization (performance profiling)

**Validation**:
- Render known scenes (Cornell box, etc.)
- Compare BDPT result to unidirectional with same depth
- Verify MIS weights sum correctly
- Check for bias or artifacts
