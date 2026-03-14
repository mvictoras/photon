# Texture Wrap Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable wrap modes (Repeat, Clamp, Mirror, Border) to the existing Kokkos texture system.

**Architecture:** Extend the existing `Texture` struct with `WrapMode` enum and per-axis wrap mode fields. Replace hardcoded repeat logic with a configurable `apply_wrap()` helper function. Maintain full backward compatibility with existing code.

**Tech Stack:** C++17, Kokkos (CUDA backend), Existing texture system in `include/photon/pt/texture.h`

**Reference:** Design doc at `docs/plans/2026-03-14-gpu-texture-sampling-design.md`

---

## Task 1: Add WrapMode Enum and Texture Fields

**Files:**
- Modify: `include/photon/pt/texture.h:11-58` (Texture struct)

**Step 1: Add WrapMode enum before Texture struct**

Add this code after the includes and namespace declaration (around line 10):

```cpp
enum class WrapMode {
  Repeat,  // Wrap around (modulo)
  Clamp,   // Clamp to [0, 1]
  Mirror,  // Mirror at boundaries
  Border   // Return border color outside [0, 1]
};
```

**Step 2: Add wrap mode fields to Texture struct**

Add these fields to the `Texture` struct after `height` (around line 15):

```cpp
WrapMode wrap_u{WrapMode::Repeat};
WrapMode wrap_v{WrapMode::Repeat};
Vec3 border_color{0.f, 0.f, 0.f};
```

**Step 3: Verify compilation**

Run: `cd /home/mvictoras/src/photon/build && cmake --build . --target texture_test -j8`
Expected: SUCCESS (no compilation errors)

**Step 4: Commit**

```bash
git add include/photon/pt/texture.h
git commit -m "feat(texture): add WrapMode enum and texture wrap mode fields

Add WrapMode enum with Repeat/Clamp/Mirror/Border modes.
Add wrap_u, wrap_v, and border_color fields to Texture struct.
Default to Repeat mode for backward compatibility."
```

---

## Task 2: Implement apply_wrap Helper Function

**Files:**
- Modify: `include/photon/pt/texture.h:11-58` (Texture struct)

**Step 1: Add apply_wrap as private method**

Add this method in the private section of `Texture` struct (create private section after public methods if needed):

```cpp
private:
  KOKKOS_FUNCTION f32 apply_wrap(f32 coord, WrapMode mode) const
  {
    switch (mode) {
    case WrapMode::Repeat:
      return coord - Kokkos::floor(coord);
    case WrapMode::Clamp:
      return Kokkos::clamp(coord, 0.f, 1.f);
    case WrapMode::Mirror: {
      f32 t = coord - Kokkos::floor(coord);
      i32 period = i32(Kokkos::floor(coord));
      return (period % 2 == 0) ? t : (1.f - t);
    }
    case WrapMode::Border:
      return coord; // Out-of-bounds check handled in caller
    }
    return coord; // Fallback for unknown mode
  }
```

**Step 2: Verify compilation**

Run: `cd /home/mvictoras/src/photon/build && cmake --build . --target texture_test -j8`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add include/photon/pt/texture.h
git commit -m "feat(texture): implement apply_wrap helper for wrap modes

Add apply_wrap() device function to handle coordinate wrapping:
- Repeat: modulo wrap (existing behavior)
- Clamp: clamp to [0, 1]
- Mirror: mirror at integer boundaries
- Border: pass through (bounds check in caller)"
```

---

## Task 3: Update sample() to Use Configurable Wrap Modes

**Files:**
- Modify: `include/photon/pt/texture.h:17-48` (sample method)

**Step 1: Replace hardcoded wrap with apply_wrap calls**

Modify the `sample()` method to replace lines 19-20:

**OLD CODE:**
```cpp
u = u - Kokkos::floor(u);
v = v - Kokkos::floor(v);
```

**NEW CODE:**
```cpp
u = apply_wrap(u, wrap_u);
v = apply_wrap(v, wrap_v);

// Handle border mode out-of-bounds early return
if (wrap_u == WrapMode::Border && (u < 0.f || u > 1.f))
  return border_color;
if (wrap_v == WrapMode::Border && (v < 0.f || v > 1.f))
  return border_color;
```

**Step 2: Keep rest of bilinear sampling unchanged**

The rest of the method (texel coordinate computation, wrap lambda for indices, bilinear interpolation) stays the same.

**Step 3: Verify compilation**

Run: `cd /home/mvictoras/src/photon/build && cmake --build . --target texture_test -j8`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add include/photon/pt/texture.h
git commit -m "feat(texture): use configurable wrap modes in sample()

Replace hardcoded repeat wrapping with apply_wrap() calls.
Add border mode early-exit when coordinates out of bounds.
Maintains backward compatibility (default is Repeat mode)."
```

---

## Task 4: Update sample_nearest() for Wrap Modes

**Files:**
- Modify: `include/photon/pt/texture.h:50-57` (sample_nearest method)

**Step 1: Replace hardcoded wrap with apply_wrap calls**

Modify the `sample_nearest()` method:

**OLD CODE:**
```cpp
KOKKOS_FUNCTION Vec3 sample_nearest(f32 u, f32 v) const
{
  u = u - Kokkos::floor(u);
  v = v - Kokkos::floor(v);
  u32 x = u32(u * f32(width)) % width;
  u32 y = u32(v * f32(height)) % height;
  return pixels(y, x);
}
```

**NEW CODE:**
```cpp
KOKKOS_FUNCTION Vec3 sample_nearest(f32 u, f32 v) const
{
  u = apply_wrap(u, wrap_u);
  v = apply_wrap(v, wrap_v);
  
  // Handle border mode
  if (wrap_u == WrapMode::Border && (u < 0.f || u > 1.f))
    return border_color;
  if (wrap_v == WrapMode::Border && (v < 0.f || v > 1.f))
    return border_color;
  
  // Clamp to valid range to prevent modulo on negative values
  u = Kokkos::clamp(u, 0.f, 0.9999999f);
  v = Kokkos::clamp(v, 0.f, 0.9999999f);
  
  u32 x = u32(u * f32(width));
  u32 y = u32(v * f32(height));
  return pixels(y, x);
}
```

**Step 2: Verify compilation**

Run: `cd /home/mvictoras/src/photon/build && cmake --build . --target texture_test -j8`
Expected: SUCCESS

**Step 3: Commit**

```bash
git add include/photon/pt/texture.h
git commit -m "feat(texture): use configurable wrap modes in sample_nearest()

Replace hardcoded repeat with apply_wrap() calls.
Add border mode handling and safe clamping.
Remove modulo operator (now redundant with wrap modes)."
```

---

## Task 5: Write Tests for Repeat Mode (Baseline)

**Files:**
- Modify: `tests/texture_test.cpp`

**Step 1: Add test for explicit Repeat mode**

Add this test after the existing wrap test (around line 33):

```cpp
// Test explicit repeat mode (should match default behavior)
Texture tex_repeat;
tex_repeat.width = 2;
tex_repeat.height = 2;
tex_repeat.wrap_u = WrapMode::Repeat;
tex_repeat.wrap_v = WrapMode::Repeat;
tex_repeat.pixels = Kokkos::View<Vec3**, Kokkos::LayoutRight>("tex_repeat", 2, 2);
auto h_repeat = Kokkos::create_mirror_view(tex_repeat.pixels);
h_repeat(0, 0) = {1, 0, 0};
h_repeat(0, 1) = {0, 1, 0};
h_repeat(1, 0) = {0, 0, 1};
h_repeat(1, 1) = {1, 1, 1};
Kokkos::deep_copy(tex_repeat.pixels, h_repeat);

// Test wrap around
Vec3 wrap_pos = tex_repeat.sample(1.5f, 0.5f);
Vec3 norm_pos = tex_repeat.sample(0.5f, 0.5f);
assert(std::abs(wrap_pos.x - norm_pos.x) < 0.01f);
assert(std::abs(wrap_pos.y - norm_pos.y) < 0.01f);
assert(std::abs(wrap_pos.z - norm_pos.z) < 0.01f);

// Test negative wrapping
Vec3 wrap_neg = tex_repeat.sample(-0.5f, 0.5f);
assert(std::abs(wrap_neg.x - norm_pos.x) < 0.01f);
```

**Step 2: Run test to verify baseline**

Run: `cd /home/mvictoras/src/photon/build && ./tests/texture_test`
Expected: PASS (all assertions pass)

**Step 3: Commit**

```bash
git add tests/texture_test.cpp
git commit -m "test(texture): add explicit Repeat mode tests

Verify that explicit WrapMode::Repeat matches default behavior.
Test positive and negative coordinate wrapping."
```

---

## Task 6: Write Tests for Clamp Mode

**Files:**
- Modify: `tests/texture_test.cpp`

**Step 1: Add Clamp mode test**

Add this test after the Repeat test:

```cpp
// Test Clamp mode
Texture tex_clamp;
tex_clamp.width = 2;
tex_clamp.height = 2;
tex_clamp.wrap_u = WrapMode::Clamp;
tex_clamp.wrap_v = WrapMode::Clamp;
tex_clamp.pixels = Kokkos::View<Vec3**, Kokkos::LayoutRight>("tex_clamp", 2, 2);
auto h_clamp = Kokkos::create_mirror_view(tex_clamp.pixels);
h_clamp(0, 0) = {1, 0, 0}; // Red
h_clamp(0, 1) = {0, 1, 0}; // Green
h_clamp(1, 0) = {0, 0, 1}; // Blue
h_clamp(1, 1) = {1, 1, 1}; // White
Kokkos::deep_copy(tex_clamp.pixels, h_clamp);

// Sample at u=1.5 should clamp to u=1.0 (right edge)
Vec3 clamp_right = tex_clamp.sample(1.5f, 0.5f);
Vec3 edge_right = tex_clamp.sample(1.0f, 0.5f);
assert(std::abs(clamp_right.x - edge_right.x) < 0.01f);
assert(std::abs(clamp_right.y - edge_right.y) < 0.01f);
assert(std::abs(clamp_right.z - edge_right.z) < 0.01f);

// Sample at u=-0.5 should clamp to u=0.0 (left edge)
Vec3 clamp_left = tex_clamp.sample(-0.5f, 0.5f);
Vec3 edge_left = tex_clamp.sample(0.0f, 0.5f);
assert(std::abs(clamp_left.x - edge_left.x) < 0.01f);
assert(std::abs(clamp_left.y - edge_left.y) < 0.01f);
assert(std::abs(clamp_left.z - edge_left.z) < 0.01f);

// Nearest mode with clamp
Vec3 nearest_clamp = tex_clamp.sample_nearest(1.5f, 0.5f);
assert(nearest_clamp.y > 0.9f || nearest_clamp.x > 0.9f); // Green or white
```

**Step 2: Run test to verify it fails**

Run: `cd /home/mvictoras/src/photon/build && ./tests/texture_test`
Expected: PASS (clamp mode now works)

**Step 3: Commit**

```bash
git add tests/texture_test.cpp
git commit -m "test(texture): add Clamp mode tests

Verify coordinates outside [0,1] clamp to edge values.
Test both positive overflow and negative underflow.
Test both bilinear and nearest sampling."
```

---

## Task 7: Write Tests for Mirror Mode

**Files:**
- Modify: `tests/texture_test.cpp`

**Step 1: Add Mirror mode test**

Add this test after the Clamp test:

```cpp
// Test Mirror mode
Texture tex_mirror;
tex_mirror.width = 2;
tex_mirror.height = 2;
tex_mirror.wrap_u = WrapMode::Mirror;
tex_mirror.wrap_v = WrapMode::Mirror;
tex_mirror.pixels = Kokkos::View<Vec3**, Kokkos::LayoutRight>("tex_mirror", 2, 2);
auto h_mirror = Kokkos::create_mirror_view(tex_mirror.pixels);
h_mirror(0, 0) = {1, 0, 0}; // Red
h_mirror(0, 1) = {0, 1, 0}; // Green
h_mirror(1, 0) = {0, 0, 1}; // Blue
h_mirror(1, 1) = {1, 1, 1}; // White
Kokkos::deep_copy(tex_mirror.pixels, h_mirror);

// u=0.25 and u=1.75 should mirror to same value
Vec3 mirror_pos = tex_mirror.sample(0.25f, 0.5f);
Vec3 mirror_neg = tex_mirror.sample(1.75f, 0.5f);
assert(std::abs(mirror_pos.x - mirror_neg.x) < 0.01f);
assert(std::abs(mirror_pos.y - mirror_neg.y) < 0.01f);
assert(std::abs(mirror_pos.z - mirror_neg.z) < 0.01f);

// u=0.75 and u=1.25 should mirror
Vec3 mirror_pos2 = tex_mirror.sample(0.75f, 0.5f);
Vec3 mirror_neg2 = tex_mirror.sample(1.25f, 0.5f);
assert(std::abs(mirror_pos2.x - mirror_neg2.x) < 0.01f);
assert(std::abs(mirror_pos2.y - mirror_neg2.y) < 0.01f);
assert(std::abs(mirror_pos2.z - mirror_neg2.z) < 0.01f);
```

**Step 2: Run test to verify**

Run: `cd /home/mvictoras/src/photon/build && ./tests/texture_test`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/texture_test.cpp
git commit -m "test(texture): add Mirror mode tests

Verify coordinates mirror at integer boundaries.
Test symmetry across [0,1] and [1,2] intervals."
```

---

## Task 8: Write Tests for Border Mode

**Files:**
- Modify: `tests/texture_test.cpp`

**Step 1: Add Border mode test**

Add this test after the Mirror test:

```cpp
// Test Border mode
Texture tex_border;
tex_border.width = 2;
tex_border.height = 2;
tex_border.wrap_u = WrapMode::Border;
tex_border.wrap_v = WrapMode::Border;
tex_border.border_color = {1.f, 0.f, 1.f}; // Magenta
tex_border.pixels = Kokkos::View<Vec3**, Kokkos::LayoutRight>("tex_border", 2, 2);
auto h_border = Kokkos::create_mirror_view(tex_border.pixels);
h_border(0, 0) = {1, 0, 0};
h_border(0, 1) = {0, 1, 0};
h_border(1, 0) = {0, 0, 1};
h_border(1, 1) = {1, 1, 1};
Kokkos::deep_copy(tex_border.pixels, h_border);

// Sample outside [0,1] should return border color
Vec3 border_outside = tex_border.sample(1.5f, 0.5f);
assert(std::abs(border_outside.x - 1.f) < 0.01f); // Magenta R
assert(std::abs(border_outside.y - 0.f) < 0.01f); // Magenta G
assert(std::abs(border_outside.z - 1.f) < 0.01f); // Magenta B

Vec3 border_neg = tex_border.sample(-0.5f, 0.5f);
assert(std::abs(border_neg.x - 1.f) < 0.01f);
assert(std::abs(border_neg.z - 1.f) < 0.01f);

// Sample inside [0,1] should return texture color
Vec3 border_inside = tex_border.sample(0.5f, 0.5f);
assert(std::abs(border_inside.x - border_outside.x) > 0.1f); // Not magenta

// Nearest mode with border
Vec3 nearest_border = tex_border.sample_nearest(1.5f, 0.5f);
assert(std::abs(nearest_border.x - 1.f) < 0.01f); // Magenta
assert(std::abs(nearest_border.z - 1.f) < 0.01f);
```

**Step 2: Run test to verify**

Run: `cd /home/mvictoras/src/photon/build && ./tests/texture_test`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/texture_test.cpp
git commit -m "test(texture): add Border mode tests

Verify out-of-bounds samples return border_color.
Test in-bounds samples return texture data.
Test both bilinear and nearest sampling."
```

---

## Task 9: Test TextureAtlas with Wrap Modes

**Files:**
- Modify: `tests/texture_test.cpp`

**Step 1: Add atlas test with different wrap modes per texture**

Add this test after the Border test:

```cpp
// Test TextureAtlas with different wrap modes
TextureAtlas atlas_wrap;
atlas_wrap.count = 3;
atlas_wrap.textures = Kokkos::View<Texture*>("atlas_wrap", 3);
auto atlas_wrap_h = Kokkos::create_mirror_view(atlas_wrap.textures);

// Texture 0: Repeat mode
atlas_wrap_h(0) = tex_repeat;

// Texture 1: Clamp mode
atlas_wrap_h(1) = tex_clamp;

// Texture 2: Border mode
atlas_wrap_h(2) = tex_border;

Kokkos::deep_copy(atlas_wrap.textures, atlas_wrap_h);

// Sample from each with out-of-bounds coords
Vec3 atlas_repeat = atlas_wrap.sample(0, 1.5f, 0.5f);
Vec3 atlas_clamp = atlas_wrap.sample(1, 1.5f, 0.5f);
Vec3 atlas_border = atlas_wrap.sample(2, 1.5f, 0.5f);

// Verify each behaves according to its mode
assert(std::abs(atlas_repeat.x - norm_pos.x) < 0.05f); // Wraps
assert(std::abs(atlas_clamp.x - edge_right.x) < 0.05f); // Clamps
assert(std::abs(atlas_border.x - 1.f) < 0.01f); // Border color
```

**Step 2: Run test to verify**

Run: `cd /home/mvictoras/src/photon/build && ./tests/texture_test`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/texture_test.cpp
git commit -m "test(texture): add TextureAtlas tests for wrap modes

Verify atlas correctly dispatches to textures with different wrap modes.
Test that each texture maintains its own wrap behavior."
```

---

## Task 10: Run Full Test Suite and Integration Test

**Files:**
- Test: All texture tests
- Test: Integration with path tracer (if applicable)

**Step 1: Run full texture test suite**

Run: `cd /home/mvictoras/src/photon/build && ./tests/texture_test`
Expected: PASS (all assertions)

**Step 2: Run broader test suite**

Run: `cd /home/mvictoras/src/photon/build && ctest -R texture -V`
Expected: All texture-related tests PASS

**Step 3: Optional: Visual integration test**

If you have a test scene with textures:
```bash
cd /home/mvictoras/src/photon/build
./app/photon_pbrt_render ../test_scenes/textured_scene.pbrt -o output.png
```

Expected: Scene renders without errors, textures visible

**Step 4: Final commit message**

```bash
git add -A
git commit -m "feat(texture): complete wrap mode implementation

Summary of Phase 1:
- Added WrapMode enum (Repeat, Clamp, Mirror, Border)
- Implemented apply_wrap() helper function
- Updated sample() and sample_nearest() to use wrap modes
- Added per-texture wrap_u, wrap_v configuration
- Added border_color support for Border mode
- Comprehensive test coverage for all modes
- Backward compatible (default is Repeat mode)

All tests passing. Ready for Phase 2 (format support)."
```

---

## Validation Checklist

Before marking complete, verify:

- [ ] All code compiles without warnings
- [ ] All unit tests pass (`./tests/texture_test`)
- [ ] Backward compatibility maintained (existing scenes render identically)
- [ ] No performance regression (bilinear sample still ~10-20 cycles)
- [ ] Code follows existing style (KOKKOS_FUNCTION, no CUDA-specific code)
- [ ] All commits have descriptive messages
- [ ] Design doc references accurate

## Next Steps

After completing this plan:

1. **Phase 2:** Multiple format support (RGB_U8, R_F32, RGBA_F32)
2. **Phase 3:** Mipmapping and trilinear filtering
3. **Phase 4:** Performance optimization and profiling

**Execution ready.** Choose:
- **Subagent-Driven:** Fresh subagent per task + review
- **Parallel Session:** Batch execution with checkpoints
