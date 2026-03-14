# GPU Texture Sampling Design for Photon Path Tracer

**Date:** 2026-03-14  
**Status:** Approved Design  
**Target:** Kokkos backend with CUDA execution space

## Overview

This document describes the design for GPU-accessible 2D texture sampling in the Photon path tracer using Kokkos. The design validates and enhances the existing Kokkos::View-based approach (already implemented in `texture.h`) while addressing requirements for:

- Storing 2D image data (float RGB) accessible from Kokkos GPU kernels
- Sampling at arbitrary (u,v) coordinates with bilinear filtering and wrap-around
- Managing multiple textures per scene (10-20 textures of varying sizes)
- Compatibility with KOKKOS_LAMBDA closures on GPU

## Problem Statement

The challenge: Kokkos kernels run on GPU but CUDA texture objects cannot be used directly inside Kokkos lambdas easily without breaking portability and introducing CUDA-specific code into the abstraction layer.

**Requirements:**
1. Device-accessible texture data in GPU memory
2. Bilinear filtering with configurable boundary modes
3. Support for 10-20 textures of varying dimensions
4. Full compatibility with `KOKKOS_FUNCTION` and `KOKKOS_LAMBDA`
5. Maintain Kokkos portability (CUDA, HIP, OpenMP backends)

## Approach Comparison

### Option A: Kokkos Views with Manual Sampling ✅ RECOMMENDED

**Implementation:** Store textures as flattened `Kokkos::View<Vec3**>` arrays with manual bilinear sampling in device code.

**Strengths:**
- Fully portable across all Kokkos backends (CUDA, HIP, OpenMP, Serial)
- No CUDA-specific code - maintains clean Kokkos abstraction
- Works perfectly in `KOKKOS_LAMBDA` - Views are designed for this
- Control over memory layout (`LayoutRight` for coalesced access)
- Already tested and working in the codebase

**Trade-offs:**
- Manual bilinear interpolation code (complexity already handled)
- No hardware texture cache (L1/L2 cache still effective)
- Slightly more GPU register pressure vs texture objects

**Status:** Already implemented in `include/photon/pt/texture.h`

### Option B: CUDA Texture Objects

**Implementation:** Create CUDA texture objects, pass handles via Kokkos Views, use `tex2D<>()` in device code.

**Strengths:**
- Hardware texture cache and filtering
- Potential performance gains for random access patterns

**Trade-offs:**
- ❌ CUDA-only - breaks portability to Embree/HIP backends
- ❌ Complex interop between `__device__` and `KOKKOS_FUNCTION`
- ❌ Dual memory management (Kokkos + CUDA allocations)
- ⚠️ Marginal performance benefit on modern GPUs with large L1/L2 caches

**Recommendation:** Not worth the portability cost.

### Option C: Hybrid Approach

**Implementation:** Default to Option A, add `KokkosCudaOptimizedBackend` with CUDA texture objects.

**Trade-offs:**
- Code duplication and maintenance burden
- Build system complexity
- Unclear performance benefit vs cost

**Recommendation:** Only if profiling shows texture access is a bottleneck.

## Selected Approach: Enhanced Kokkos Views (Option A)

We enhance the existing implementation with:

1. **Multiple texture format support** - RGB_F32, RGBA_F32, R_F32, RGB_U8
2. **Configurable wrap modes** - Repeat, Clamp, Mirror, Border
3. **Additional filtering options** - Nearest, Bilinear, Trilinear (with mipmaps)
4. **Optimized memory layout** - Packed storage, optional compression

## Architecture

### Data Structures

```cpp
// Texture format enumeration
enum class TextureFormat {
  RGB_F32,   // 3x float32 per pixel (current implementation)
  RGBA_F32,  // 4x float32 per pixel
  R_F32,     // 1x float32 per pixel (grayscale)
  RGB_U8     // 3x uint8 per pixel (packed, LDR)
};

// Boundary handling modes
enum class WrapMode {
  Repeat,  // Wrap around (current implementation)
  Clamp,   // Clamp to [0,1]
  Mirror,  // Mirror at boundaries
  Border   // Return border color outside [0,1]
};

// Enhanced texture structure
struct Texture {
  // Pixel storage (existing)
  Kokkos::View<Vec3**, Kokkos::LayoutRight> pixels;
  u32 width{0};
  u32 height{0};
  
  // New: format and wrap mode metadata
  TextureFormat format{TextureFormat::RGB_F32};
  WrapMode wrap_u{WrapMode::Repeat};
  WrapMode wrap_v{WrapMode::Repeat};
  Vec3 border_color{0.f, 0.f, 0.f};
  
  // Sampling methods
  KOKKOS_FUNCTION Vec3 sample(f32 u, f32 v) const;
  KOKKOS_FUNCTION Vec3 sample_nearest(f32 u, f32 v) const;
  
private:
  KOKKOS_FUNCTION f32 apply_wrap(f32 coord, WrapMode mode) const;
};

// Texture atlas (existing)
struct TextureAtlas {
  Kokkos::View<Texture*> textures;
  u32 count{0};
  
  KOKKOS_FUNCTION Vec3 sample(i32 tex_id, f32 u, f32 v) const;
};
```

### Memory Layout

**Current implementation uses:** `Kokkos::View<Vec3**, Kokkos::LayoutRight>`

- Row-major storage: `pixels(row, col)` → memory offset = `row * width + col`
- Cache-friendly for typical UV access patterns in ray tracing
- `LayoutRight` ensures coalesced memory access on GPU

**Memory management workflow:**
1. Load image on host using `image_loader` (TGA/PNG/JPEG support exists)
2. Allocate `Kokkos::View` with host mirror
3. Copy pixel data to host mirror
4. `Kokkos::deep_copy()` to device (GPU)
5. Store `Texture` metadata in `TextureAtlas`

**For 10-20 textures:**
- Each texture owns its own `Kokkos::View` (no shared backing storage)
- Atlas stores array of `Texture` structs (metadata only, ~64 bytes each)
- Total GPU memory = sum of all texture pixel data + negligible overhead

### Sampling Implementation

**Bilinear filtering (already implemented):**

```cpp
KOKKOS_FUNCTION Vec3 Texture::sample(f32 u, f32 v) const {
  // Apply wrap mode
  u = u - Kokkos::floor(u);  // Current: hardcoded repeat
  v = v - Kokkos::floor(v);
  
  // Compute texel coordinates
  f32 fx = u * f32(width) - 0.5f;
  f32 fy = v * f32(height) - 0.5f;
  
  i32 ix = i32(Kokkos::floor(fx));
  i32 iy = i32(Kokkos::floor(fy));
  f32 tx = fx - Kokkos::floor(fx);
  f32 ty = fy - Kokkos::floor(fy);
  
  // Wrap indices
  auto wrap = [](i32 i, u32 sz) -> u32 {
    i32 s = i32(sz);
    return u32(((i % s) + s) % s);
  };
  
  u32 x0 = wrap(ix, width);
  u32 x1 = wrap(ix + 1, width);
  u32 y0 = wrap(iy, height);
  u32 y1 = wrap(iy + 1, height);
  
  // Fetch 2x2 neighborhood
  Vec3 c00 = pixels(y0, x0);
  Vec3 c10 = pixels(y0, x1);
  Vec3 c01 = pixels(y1, x0);
  Vec3 c11 = pixels(y1, x1);
  
  // Bilinear interpolation
  Vec3 top = c00 * (1.f - tx) + c10 * tx;
  Vec3 bot = c01 * (1.f - tx) + c11 * tx;
  return top * (1.f - ty) + bot * ty;
}
```

**Enhancement: Configurable wrap modes**

Replace hardcoded repeat with:

```cpp
KOKKOS_FUNCTION f32 Texture::apply_wrap(f32 coord, WrapMode mode) const {
  switch(mode) {
    case WrapMode::Repeat:
      return coord - Kokkos::floor(coord);
    case WrapMode::Clamp:
      return Kokkos::clamp(coord, 0.f, 1.f);
    case WrapMode::Mirror: {
      f32 t = coord - Kokkos::floor(coord);
      return (int(Kokkos::floor(coord)) % 2) ? (1.f - t) : t;
    }
    case WrapMode::Border:
      return coord; // Handled in sample() with bounds check
  }
  return coord;
}
```

## Integration with Materials

**Current integration (already working):**

```cpp
struct Material {
  Vec3 base_color{0.8f, 0.8f, 0.8f};
  f32 metallic{0.f};
  f32 roughness{0.5f};
  // ...
  i32 base_color_tex{-1};   // Texture ID or -1
  i32 normal_tex{-1};
  i32 roughness_tex{-1};
  i32 metallic_tex{-1};
  i32 emission_tex{-1};
};
```

**Usage in path tracing kernel:**

```cpp
Kokkos::parallel_for("shade_rays", num_rays, KOKKOS_LAMBDA(u32 i) {
  const HitResult& hit = hits(i);
  const Material& mat = materials(hit.material_id);
  const TextureAtlas& textures = scene.textures;
  
  // Sample base color (default or texture)
  Vec3 albedo = mat.base_color;
  if (mat.base_color_tex >= 0) {
    albedo = textures.sample(mat.base_color_tex, hit.uv.x, hit.uv.y);
  }
  
  // Sample normal map and transform to world space
  Vec3 normal = hit.shading_normal;
  if (mat.normal_tex >= 0) {
    Vec3 normal_map = textures.sample(mat.normal_tex, hit.uv.x, hit.uv.y);
    // Convert from [0,1] to [-1,1] tangent space
    normal_map = normal_map * 2.f - Vec3(1.f, 1.f, 1.f);
    // Apply tangent-space to world-space transform
    // normal = TBN * normal_map; (requires tangent vectors)
  }
  
  // Sample roughness/metallic from textures
  f32 roughness = mat.roughness;
  if (mat.roughness_tex >= 0) {
    roughness = textures.sample(mat.roughness_tex, hit.uv.x, hit.uv.y).x;
  }
  
  // Continue with Disney BSDF evaluation...
});
```

## Performance Considerations

### GPU Memory Access

**Coalesced access:** `LayoutRight` ensures adjacent threads accessing nearby UVs load consecutive memory.

**Cache efficiency:** 
- L1 cache (128KB on A100): Fits ~42K Vec3 pixels
- L2 cache (40MB on A100): Fits entire texture atlas for most scenes
- Bilinear sampling: 4 pixel fetches likely hit L1 cache

**Expected performance:**
- Bilinear sample: ~10-20 GPU cycles (L1 hit)
- Compare to CUDA texture object: ~10-15 cycles
- Difference: negligible in path tracing context (BSDF eval dominates)

### Memory Footprint

**Example scene (10 textures):**
- 5x 2K textures (2048x2048): 5 × 12MB = 60MB
- 3x 1K textures (1024x1024): 3 × 3MB = 9MB
- 2x 512 textures (512x512): 2 × 768KB = 1.5MB
- **Total:** ~70MB GPU memory

Modern GPUs (RTX 4090: 24GB, A100: 80GB) handle this easily.

### Optimization Opportunities

1. **Texture compression** (future):
   - BC6H for HDR textures: 8:1 compression
   - BC7 for LDR textures: 4:1 compression
   - Requires on-GPU decompression

2. **Mipmapping** (future):
   - Precompute mip levels on load
   - Select level based on ray differentials
   - Better cache performance for minification

3. **Texture streaming** (future):
   - Load textures on-demand for very large scenes
   - Least-recently-used eviction policy

## Testing Strategy

**Unit tests (extend `tests/texture_test.cpp`):**
- Wrap mode correctness (repeat, clamp, mirror, border)
- Format conversion accuracy
- Bilinear filtering precision
- Atlas lookup with invalid IDs

**Integration tests:**
- Load real PBRT scenes with textured materials
- Compare against reference renderer (PBRT-v4)
- Visual validation of normal mapping
- Performance benchmarking vs current implementation

**Regression tests:**
- Ensure existing scenes render identically
- Validate backward compatibility

## Implementation Phases

### Phase 1: Wrap Mode Support (Small)
- Add `WrapMode` enum and `wrap_u/wrap_v` fields
- Implement `apply_wrap()` helper
- Update `sample()` to use configurable modes
- Add tests

### Phase 2: Multiple Format Support (Medium)
- Add `TextureFormat` enum
- Support RGB_U8 packed format (memory savings)
- Support single-channel R_F32 (grayscale)
- Add format-aware loading in `image_loader`

### Phase 3: Enhanced Filtering (Medium)
- Add mipmap storage to `Texture`
- Implement trilinear filtering
- Add mip level selection based on derivatives
- Benchmark performance

### Phase 4: Optimization (Optional)
- Profile texture access patterns
- Experiment with texture compression
- Optimize memory layout for large atlases

## Alternatives Considered

**Why not CUDA texture objects?**
- Breaks Kokkos portability (no Embree, HIP backends)
- Complex interop with `KOKKOS_FUNCTION`
- Marginal performance benefit on modern GPUs
- Harder to debug and maintain

**Why not bake textures to per-triangle colors?**
- Too coarse - loses detail from textures
- Large memory overhead for high-res geometry
- No support for normal/roughness mapping
- Current approach mentioned as "too coarse"

## Success Criteria

1. ✅ Textures accessible from `KOKKOS_LAMBDA` on GPU
2. ✅ Bilinear filtering with wrap-around working
3. ✅ Support for 10-20 textures without memory issues
4. ✅ Maintains Kokkos portability across backends
5. ✅ Performance comparable to CUDA texture objects (within 5%)
6. ✅ Backward compatible with existing scenes

## References

- Existing implementation: `include/photon/pt/texture.h`
- Kokkos Views documentation: https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html
- PBRT texture chapter: https://pbr-book.org/3ed-2018/Texture
- Recent commit: `c73a484 feat: add texture support for pbrt scenes`

## Conclusion

The current Kokkos::View-based approach (Option A) is the correct design choice for the Photon path tracer. It provides full portability, clean integration with Kokkos, and performance comparable to CUDA-specific solutions. The proposed enhancements add flexibility (wrap modes, formats) while maintaining the core architecture's strengths.

No fundamental changes needed - the existing implementation validates this design. Future work focuses on incremental improvements (mipmapping, compression) rather than architectural changes.
