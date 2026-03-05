# Photon Path Tracer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the Photon ANARI+Kokkos path tracer with modern rendering features and hardware-accelerated ray-tracing backends.

**Architecture:** Wavefront path tracing integrator with pluggable RayBackend (Embree/OptiX/HIP RT/Kokkos fallback). Disney Principled BSDF with SSS for materials. All shading stays in Kokkos for portability.

**Tech Stack:** C++23, Kokkos 5.0.1, ANARI SDK 0.15.0, Embree 4, OptiX 7+, HIP RT, Intel OIDN, stb_image

**Design Doc:** `docs/plans/2026-03-05-photon-pathtracer-design.md`

---

## Phase 1: Foundation — Math, Materials, Sampling Infrastructure

These tasks build the math and data structures everything else depends on.

---

### Task 1: Extend Vec3 math and add Vec2, Mat4, color utilities

**Files:**
- Modify: `include/photon/pt/math.h`
- Create: `include/photon/pt/math_vec2.h`
- Create: `include/photon/pt/math_mat4.h`
- Create: `include/photon/pt/color.h`
- Test: `tests/math_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/math_test.cpp
#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/math_mat4.h"
#include "photon/pt/color.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    // Vec3 extensions
    Vec3 a{1, 2, 3};
    assert(max_component(a) == 3.f);
    assert(min_component(a) == 1.f);
    Vec3 neg = -a;
    assert(neg.x == -1.f && neg.y == -2.f && neg.z == -3.f);
    Vec3 d = a / 2.f;
    assert(std::abs(d.x - 0.5f) < 1e-5f);

    // Vec2
    Vec2 v2{0.5f, 0.3f};
    assert(std::abs(v2.x - 0.5f) < 1e-5f);

    // Mat4 identity
    Mat4 m = Mat4::identity();
    Vec3 p{1, 2, 3};
    Vec3 tp = m.transform_point(p);
    assert(std::abs(tp.x - 1.f) < 1e-5f);

    // Mat4 translation
    Mat4 t = Mat4::translate({10, 20, 30});
    Vec3 tp2 = t.transform_point(p);
    assert(std::abs(tp2.x - 11.f) < 1e-5f);

    // Color utilities
    Vec3 linear{0.5f, 0.5f, 0.5f};
    Vec3 srgb = linear_to_srgb(linear);
    assert(srgb.x > 0.7f); // gamma curve
    Vec3 back = srgb_to_linear(srgb);
    assert(std::abs(back.x - 0.5f) < 0.01f);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

Run: `cmake --build build -j && ctest --test-dir build -R photon_math_test`
Expected: FAIL — functions not defined

**Step 3: Implement**

Add to `math.h`:
- `operator-()` (unary negation)
- `operator/(Vec3, f32)`
- `operator+=(Vec3&, Vec3)` already exists
- `max_component(Vec3)`, `min_component(Vec3)`
- `reflect(Vec3 I, Vec3 N)` — reflection
- `refract(Vec3 I, Vec3 N, f32 eta)` — Snell's law refraction
- `lerp(Vec3 a, Vec3 b, f32 t)`

Create `math_vec2.h`:
```cpp
struct Vec2 {
  f32 x{0.f}, y{0.f};
  KOKKOS_FUNCTION Vec2() = default;
  KOKKOS_FUNCTION Vec2(f32 xx, f32 yy) : x(xx), y(yy) {}
};
```

Create `math_mat4.h`:
```cpp
struct Mat4 {
  f32 m[4][4]{};
  static KOKKOS_FUNCTION Mat4 identity();
  static KOKKOS_FUNCTION Mat4 translate(const Vec3& t);
  static KOKKOS_FUNCTION Mat4 scale(const Vec3& s);
  KOKKOS_FUNCTION Vec3 transform_point(const Vec3& p) const;
  KOKKOS_FUNCTION Vec3 transform_direction(const Vec3& d) const;
  KOKKOS_FUNCTION Vec3 transform_normal(const Vec3& n) const; // inverse transpose
  KOKKOS_FUNCTION Mat4 inverse() const;
  KOKKOS_FUNCTION Mat4 operator*(const Mat4& b) const;
};
```

Create `color.h`:
```cpp
KOKKOS_FUNCTION Vec3 linear_to_srgb(Vec3 c);
KOKKOS_FUNCTION Vec3 srgb_to_linear(Vec3 c);
KOKKOS_FUNCTION f32 luminance(Vec3 c); // 0.2126R + 0.7152G + 0.0722B
KOKKOS_FUNCTION Vec3 aces_tonemap(Vec3 c);
```

**Step 4: Run tests**

Run: `cmake --build build -j && ctest --test-dir build -R photon_math_test`
Expected: PASS

**Step 5: Commit**

```bash
git add include/photon/pt/math.h include/photon/pt/math_vec2.h \
        include/photon/pt/math_mat4.h include/photon/pt/color.h \
        tests/math_test.cpp tests/CMakeLists.txt
git commit -m "feat: add Vec2, Mat4, color utilities, extend Vec3 math"
```

---

### Task 2: Sampling infrastructure — Sobol, hemisphere, MIS utilities

**Files:**
- Create: `include/photon/pt/sampling.h`
- Create: `src/photon/pt/sampling.cpp`
- Test: `tests/sampling_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/sampling_test.cpp
#include "photon/pt/sampling.h"
#include "photon/pt/rng.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    Rng rng(42);

    // Cosine-weighted hemisphere sampling
    Vec3 n{0, 1, 0};
    auto [dir, pdf] = sample_cosine_hemisphere(n, rng);
    assert(dot(dir, n) >= 0.f); // above hemisphere
    assert(pdf > 0.f);

    // Power heuristic MIS
    f32 w = power_heuristic(1, 0.5f, 1, 0.5f);
    assert(std::abs(w - 0.5f) < 1e-5f); // equal PDFs → 0.5

    // Cosine hemisphere PDF
    f32 cos_theta = dot(dir, n);
    f32 expected_pdf = cos_theta / 3.14159265f;
    assert(std::abs(pdf - expected_pdf) < 0.01f);

    // Sample uniform sphere
    Vec3 us = sample_uniform_sphere(rng);
    assert(std::abs(length(us) - 1.f) < 0.01f);

    // Sample uniform triangle
    auto [bu, bv] = sample_uniform_triangle(rng);
    assert(bu >= 0.f && bv >= 0.f && (bu + bv) <= 1.f);

    // Concentric disk mapping
    Vec2 disk = sample_concentric_disk(rng);
    assert(disk.x * disk.x + disk.y * disk.y <= 1.f + 1e-5f);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `sampling.h`**

```cpp
#pragma once
#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/rng.h"

namespace photon::pt {

// MIS power heuristic (beta=2)
KOKKOS_FUNCTION inline f32 power_heuristic(int nf, f32 fPdf, int ng, f32 gPdf) {
  f32 f = nf * fPdf;
  f32 g = ng * gPdf;
  return (f * f) / (f * f + g * g);
}

// Cosine-weighted hemisphere sampling (returns direction + pdf)
struct SampleResult { Vec3 direction; f32 pdf; };
KOKKOS_FUNCTION SampleResult sample_cosine_hemisphere(const Vec3& normal, Rng& rng);

// Uniform sphere sampling
KOKKOS_FUNCTION Vec3 sample_uniform_sphere(Rng& rng);

// Uniform triangle sampling (returns barycentric u,v)
struct BaryResult { f32 u; f32 v; };
KOKKOS_FUNCTION BaryResult sample_uniform_triangle(Rng& rng);

// Concentric disk mapping (Shirley-Chiu)
KOKKOS_FUNCTION Vec2 sample_concentric_disk(Rng& rng);

// Build orthonormal basis from a single normal
KOKKOS_FUNCTION void onb_from_normal(const Vec3& n, Vec3& tangent, Vec3& bitangent);

// Cosine hemisphere PDF
KOKKOS_FUNCTION inline f32 cosine_hemisphere_pdf(f32 cos_theta) {
  return cos_theta / 3.14159265f;
}

constexpr f32 PI = 3.14159265358979323846f;
constexpr f32 INV_PI = 1.f / PI;
constexpr f32 TWO_PI = 2.f * PI;
constexpr f32 INV_TWO_PI = 1.f / TWO_PI;

} // namespace photon::pt
```

**Step 4: Run tests**
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat: add sampling infrastructure — cosine hemisphere, MIS, disk, ONB"
```

---

### Task 3: Extend TriangleMesh with normals, UVs, material IDs

**Files:**
- Modify: `include/photon/pt/geom/triangle_mesh.h`
- Modify: `include/photon/pt/geom/mesh_intersector.h`
- Modify: `include/photon/pt/geom/triangle_intersect.h`
- Test: `tests/mesh_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/mesh_test.cpp — test that MeshHit now includes UV + smooth normal
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/geom/mesh_intersector.h"
#include "photon/pt/bvh/bvh.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    // Build a single triangle with normals and UVs
    TriangleMesh mesh;
    mesh.positions = Kokkos::View<Vec3*>("pos", 3);
    mesh.indices = Kokkos::View<u32*>("idx", 3);
    mesh.normals = Kokkos::View<Vec3*>("nor", 3);
    mesh.texcoords = Kokkos::View<Vec2*>("uv", 3);
    mesh.material_ids = Kokkos::View<u32*>("mat", 1);

    auto pos_h = Kokkos::create_mirror_view(mesh.positions);
    auto idx_h = Kokkos::create_mirror_view(mesh.indices);
    auto nor_h = Kokkos::create_mirror_view(mesh.normals);
    auto uv_h = Kokkos::create_mirror_view(mesh.texcoords);
    auto mat_h = Kokkos::create_mirror_view(mesh.material_ids);

    pos_h(0) = {-1, -1, -2}; pos_h(1) = {1, -1, -2}; pos_h(2) = {0, 1, -2};
    idx_h(0) = 0; idx_h(1) = 1; idx_h(2) = 2;
    nor_h(0) = {0, 0, 1}; nor_h(1) = {0, 0, 1}; nor_h(2) = {0, 0, 1};
    uv_h(0) = {0, 0}; uv_h(1) = {1, 0}; uv_h(2) = {0.5f, 1};
    mat_h(0) = 7;

    Kokkos::deep_copy(mesh.positions, pos_h);
    Kokkos::deep_copy(mesh.indices, idx_h);
    Kokkos::deep_copy(mesh.normals, nor_h);
    Kokkos::deep_copy(mesh.texcoords, uv_h);
    Kokkos::deep_copy(mesh.material_ids, mat_h);

    auto bvh = Bvh::build_cpu(mesh);
    Ray ray{{0, 0, 0}, {0, 0, -1}};
    MeshHit hit = intersect_mesh_bvh(mesh, bvh, ray, 1e-3f, 1e30f);

    assert(hit.hit);
    assert(std::abs(hit.t - 2.f) < 0.01f);
    assert(hit.material_id == 7u);
    // UV should be interpolated at hit point
    assert(hit.uv.x >= 0.f && hit.uv.x <= 1.f);
    assert(hit.uv.y >= 0.f && hit.uv.y <= 1.f);
    // Smooth normal should be {0,0,1}
    assert(std::abs(hit.shading_normal.z - 1.f) < 0.01f);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

Extend `TriangleMesh`:
```cpp
struct TriangleMesh {
  Kokkos::View<Vec3*> positions;
  Kokkos::View<u32*>  indices;
  Kokkos::View<Vec3*> normals;       // per-vertex (optional)
  Kokkos::View<Vec2*> texcoords;     // per-vertex (optional)
  Kokkos::View<u32*>  material_ids;  // per-primitive (optional)
  Kokkos::View<Vec3*> albedo_per_prim; // legacy, keep for compat
  // ...
};
```

Extend `MeshHit`:
```cpp
struct MeshHit {
  f32 t{0.f};
  Vec3 p;
  Vec3 n;              // geometric normal
  Vec3 shading_normal; // interpolated vertex normal (or geometric if no normals)
  Vec2 uv;             // interpolated texture coordinates
  u32 prim_id{0};
  u32 geom_id{0};
  u32 material_id{0};
  bool hit{false};
};
```

Update `intersect_mesh_bvh()` to interpolate normals and UVs using barycentric coordinates from `TriHit::u, v`.

**Step 4: Run tests**
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat: extend TriangleMesh with normals, UVs, material IDs"
```

---

### Task 4: Disney Principled BSDF — material structure and evaluation

**Files:**
- Rewrite: `include/photon/pt/material.h`
- Create: `include/photon/pt/disney_bsdf.h`
- Create: `src/photon/pt/disney_bsdf.cpp`
- Test: `tests/disney_bsdf_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/disney_bsdf_test.cpp
#include "photon/pt/disney_bsdf.h"
#include "photon/pt/material.h"
#include "photon/pt/rng.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    Rng rng(42);

    // Default material = diffuse gray
    Material mat;
    Vec3 wo{0, 0, 1}; // outgoing dir (toward camera)
    Vec3 n{0, 0, 1};  // surface normal

    // Sample BSDF
    BsdfSample s = disney_bsdf_sample(mat, wo, n, n, rng);
    assert(s.pdf > 0.f);
    assert(dot(s.wi, n) > 0.f); // sampled direction in upper hemisphere
    assert(s.f.x > 0.f);        // non-zero BSDF value

    // Evaluate BSDF
    Vec3 f = disney_bsdf_eval(mat, wo, s.wi, n, n);
    assert(f.x >= 0.f); // non-negative

    // PDF
    f32 pdf = disney_bsdf_pdf(mat, wo, s.wi, n, n);
    assert(pdf > 0.f);

    // Metallic material should have no diffuse
    Material metal;
    metal.metallic = 1.f;
    metal.roughness = 0.1f;
    BsdfSample sm = disney_bsdf_sample(metal, wo, n, n, rng);
    assert(sm.pdf > 0.f);

    // Glass material
    Material glass;
    glass.transmission = 1.f;
    glass.ior = 1.5f;
    glass.roughness = 0.0f;
    BsdfSample sg = disney_bsdf_sample(glass, wo, n, n, rng);
    assert(sg.pdf > 0.f);

    // Emissive material
    Material emissive;
    emissive.emission = {10.f, 8.f, 5.f};
    emissive.emission_strength = 1.f;
    Vec3 Le = material_emission(emissive);
    assert(Le.x == 10.f);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

Rewrite `material.h` with Disney Principled parameters:
```cpp
struct Material {
  Vec3 base_color{0.8f, 0.8f, 0.8f};
  f32  metallic{0.f};
  f32  roughness{0.5f};
  f32  ior{1.5f};
  f32  transmission{0.f};
  f32  specular{0.5f};
  f32  clearcoat{0.f};
  f32  clearcoat_roughness{0.03f};
  Vec3 emission{0.f, 0.f, 0.f};
  f32  emission_strength{0.f};
  f32  subsurface{0.f};
  Vec3 subsurface_color{1.f, 1.f, 1.f};
  f32  subsurface_radius{1.f};
  i32  base_color_tex{-1};
  i32  normal_tex{-1};
  i32  roughness_tex{-1};
  i32  metallic_tex{-1};
  i32  emission_tex{-1};
};
```

Create `disney_bsdf.h` with:
```cpp
struct BsdfSample {
  Vec3 wi;        // sampled direction
  Vec3 f;         // BSDF value
  f32  pdf;       // probability density
  bool is_specular{false};
};

KOKKOS_FUNCTION BsdfSample disney_bsdf_sample(
    const Material& mat, const Vec3& wo, const Vec3& n,
    const Vec3& shading_n, Rng& rng);

KOKKOS_FUNCTION Vec3 disney_bsdf_eval(
    const Material& mat, const Vec3& wo, const Vec3& wi,
    const Vec3& n, const Vec3& shading_n);

KOKKOS_FUNCTION f32 disney_bsdf_pdf(
    const Material& mat, const Vec3& wo, const Vec3& wi,
    const Vec3& n, const Vec3& shading_n);

KOKKOS_FUNCTION Vec3 material_emission(const Material& mat);
```

Implementation in `disney_bsdf.cpp`:
- GGX/Trowbridge-Reitz NDF: `D(h) = a^2 / (pi * ((n·h)^2 * (a^2 - 1) + 1)^2)`
- Smith G1: `G1(v) = 2(n·v) / ((n·v) + sqrt(a^2 + (1-a^2)*(n·v)^2))`
- Fresnel Schlick: `F(cos) = F0 + (1-F0) * (1-cos)^5`
- Visible normal sampling (VNDF) for GGX importance sampling
- Lobe selection: diffuse weight = `(1-metallic)*(1-transmission)`, specular weight = 1, transmission weight = `transmission*(1-metallic)`

**Step 4: Run tests**
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat: add Disney Principled BSDF with GGX microfacets"
```

---

### Task 5: Light types and environment map with importance sampling

**Files:**
- Create: `include/photon/pt/light.h`
- Create: `include/photon/pt/environment_map.h`
- Create: `src/photon/pt/environment_map.cpp`
- Test: `tests/light_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/light_test.cpp
#include "photon/pt/light.h"
#include "photon/pt/environment_map.h"
#include "photon/pt/rng.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    Rng rng(42);

    // Point light sampling
    Light point;
    point.type = LightType::Point;
    point.position = {0, 5, 0};
    point.color = {1, 1, 1};
    point.intensity = 100.f;

    Vec3 hit_pos{0, 0, 0};
    auto ls = sample_light(point, hit_pos, rng);
    assert(ls.pdf > 0.f);
    assert(ls.Li.x > 0.f); // some radiance
    assert(std::abs(length(ls.wi) - 1.f) < 0.01f); // unit direction

    // Directional light
    Light dir;
    dir.type = LightType::Directional;
    dir.direction = normalize(Vec3{0, -1, 0});
    dir.color = {1, 0.9f, 0.8f};
    dir.intensity = 5.f;

    auto ds = sample_light(dir, hit_pos, rng);
    assert(ds.pdf > 0.f);

    // Create a simple 2x1 environment map (left=bright, right=dark)
    EnvironmentMap env;
    env.width = 2; env.height = 1;
    env.pixels = Kokkos::View<Vec3**, Kokkos::LayoutRight>("env", 1, 2);
    auto env_h = Kokkos::create_mirror_view(env.pixels);
    env_h(0, 0) = {10.f, 10.f, 10.f}; // bright
    env_h(0, 1) = {0.1f, 0.1f, 0.1f}; // dark
    Kokkos::deep_copy(env.pixels, env_h);
    env.build_cdf();

    // Sample should prefer bright region
    int bright_count = 0;
    for (int i = 0; i < 100; ++i) {
      f32 pdf;
      Vec3 d = env.sample_direction(rng, pdf);
      assert(pdf > 0.f);
      Vec3 L = env.evaluate(d);
      if (L.x > 5.f) bright_count++;
    }
    assert(bright_count > 50); // most samples toward bright region
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

`light.h`:
```cpp
enum class LightType : u32 { Point, Directional, Spot, Area, Environment };

struct Light {
  LightType type;
  Vec3 position;
  Vec3 direction;
  Vec3 color{1, 1, 1};
  f32  intensity{1.f};
  f32  spot_angle{0.5f};
  f32  spot_falloff{0.1f};
  u32  mesh_prim_begin{0};
  u32  mesh_prim_count{0};
  f32  area{0.f}; // for area lights
};

struct LightSample {
  Vec3 wi;       // direction to light
  Vec3 Li;       // incoming radiance
  f32  pdf;      // sampling PDF
  f32  dist;     // distance to light
  bool is_delta; // point/directional = delta distribution
};

KOKKOS_FUNCTION LightSample sample_light(const Light& light,
    const Vec3& hit_pos, Rng& rng);
```

`environment_map.h`:
```cpp
struct EnvironmentMap {
  Kokkos::View<Vec3**, Kokkos::LayoutRight> pixels;
  Kokkos::View<f32*>  marginal_cdf;
  Kokkos::View<f32**> conditional_cdf;
  u32 width{0}, height{0};

  void build_cdf(); // host-side CDF construction
  KOKKOS_FUNCTION Vec3 sample_direction(Rng& rng, f32& pdf) const;
  KOKKOS_FUNCTION Vec3 evaluate(const Vec3& direction) const;
  KOKKOS_FUNCTION f32  pdf(const Vec3& direction) const;
};
```

CDF construction uses standard 2D hierarchical approach:
1. For each row, build conditional CDF (cumulative sum of luminance)
2. Build marginal CDF from row totals
3. Sampling: sample marginal CDF for row, then conditional CDF for column

**Step 4: Run tests**
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat: add light types and environment map with importance sampling"
```

---

### Task 6: Texture system with bilinear sampling

**Files:**
- Create: `include/photon/pt/texture.h`
- Create: `src/photon/pt/texture.cpp`
- Test: `tests/texture_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`
- Modify: `cmake/Dependencies.cmake` (add stb_image)

**Step 1: Write the failing test**

```cpp
// tests/texture_test.cpp
#include "photon/pt/texture.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    // Create 2x2 texture
    Texture tex;
    tex.width = 2; tex.height = 2;
    tex.pixels = Kokkos::View<Vec3**, Kokkos::LayoutRight>("tex", 2, 2);
    auto h = Kokkos::create_mirror_view(tex.pixels);
    h(0,0) = {1,0,0}; h(0,1) = {0,1,0};
    h(1,0) = {0,0,1}; h(1,1) = {1,1,1};
    Kokkos::deep_copy(tex.pixels, h);

    // Exact corner should return corner color
    Vec3 c00 = tex.sample(0.f, 0.f);
    assert(std::abs(c00.x - 1.f) < 0.1f); // red

    // Center should be average of all 4
    Vec3 center = tex.sample(0.5f, 0.5f);
    assert(center.x > 0.3f && center.x < 0.7f); // mixed

    // TextureAtlas lookup
    TextureAtlas atlas;
    atlas.textures = Kokkos::View<Texture*>("atlas", 1);
    auto atlas_h = Kokkos::create_mirror_view(atlas.textures);
    atlas_h(0) = tex;
    Kokkos::deep_copy(atlas.textures, atlas_h);

    // Sample through atlas
    Vec3 s = atlas.sample(0, 0.f, 0.f);
    assert(std::abs(s.x - 1.f) < 0.1f);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

```cpp
// include/photon/pt/texture.h
struct Texture {
  Kokkos::View<Vec3**, Kokkos::LayoutRight> pixels;
  u32 width{0}, height{0};

  KOKKOS_FUNCTION Vec3 sample(f32 u, f32 v) const; // bilinear
  KOKKOS_FUNCTION Vec3 sample_nearest(f32 u, f32 v) const;
};

struct TextureAtlas {
  Kokkos::View<Texture*> textures;
  u32 count{0};

  KOKKOS_FUNCTION Vec3 sample(i32 tex_id, f32 u, f32 v) const;
};

// Host-side: load from file
Texture load_texture(const char* path); // uses stb_image
```

Add stb_image to Dependencies.cmake:
```cmake
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG master
)
FetchContent_MakeAvailable(stb)
```

**Step 4: Run tests**
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat: add texture system with bilinear sampling and stb_image loading"
```

---

## Phase 2: Ray Backend Abstraction

---

### Task 7: RayBackend interface and Kokkos fallback backend

**Files:**
- Create: `include/photon/pt/backend/ray_backend.h`
- Create: `include/photon/pt/backend/kokkos_backend.h`
- Create: `src/photon/pt/backend/kokkos_backend.cpp`
- Create: `src/photon/pt/backend/backend_factory.cpp`
- Create: `include/photon/pt/backend/backend_factory.h`
- Test: `tests/backend_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/backend_test.cpp
#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/backend/kokkos_backend.h"
#include "photon/pt/backend/backend_factory.h"
#include "photon/pt/scene/builder.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    auto scene = SceneBuilder::make_two_quads();

    // Create Kokkos backend
    auto backend = std::make_unique<KokkosBackend>();
    assert(std::strcmp(backend->name(), "kokkos") == 0);

    backend->build_accel(scene);

    // Trace a single ray that should hit the quad
    RayBatch rays;
    rays.count = 1;
    rays.origins = Kokkos::View<Vec3*>("org", 1);
    rays.directions = Kokkos::View<Vec3*>("dir", 1);
    rays.tmin = Kokkos::View<f32*>("tmin", 1);
    rays.tmax = Kokkos::View<f32*>("tmax", 1);

    auto org_h = Kokkos::create_mirror_view(rays.origins);
    auto dir_h = Kokkos::create_mirror_view(rays.directions);
    auto tmin_h = Kokkos::create_mirror_view(rays.tmin);
    auto tmax_h = Kokkos::create_mirror_view(rays.tmax);
    org_h(0) = {0, 0, 2};
    dir_h(0) = {0, 0, -1};
    tmin_h(0) = 1e-3f;
    tmax_h(0) = 1e30f;
    Kokkos::deep_copy(rays.origins, org_h);
    Kokkos::deep_copy(rays.directions, dir_h);
    Kokkos::deep_copy(rays.tmin, tmin_h);
    Kokkos::deep_copy(rays.tmax, tmax_h);

    HitBatch hits;
    hits.count = 1;
    hits.hits = Kokkos::View<HitResult*>("hits", 1);

    backend->trace_closest(rays, hits);

    auto hits_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hits.hits);
    assert(hits_h(0).hit);
    assert(std::abs(hits_h(0).t - 4.5f) < 0.1f); // quad is at z=-2.5

    // Test occlusion query
    Kokkos::View<u32*> occluded("occ", 1);
    backend->trace_occluded(rays, occluded);
    auto occ_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, occluded);
    assert(occ_h(0) == 1u); // should be occluded

    // Factory should return kokkos backend as fallback
    auto best = create_best_backend();
    assert(best != nullptr);
    assert(std::strlen(best->name()) > 0);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

`ray_backend.h` — abstract interface as described in design doc.

`kokkos_backend.h/cpp` — wraps existing BVH build + `intersect_mesh_bvh()` into the RayBackend interface. Uses `Kokkos::parallel_for` to trace each ray in the batch.

`backend_factory.cpp` — probes hardware, returns best available backend. For now, always returns `KokkosBackend` (Embree/OptiX/HIPRT added in later tasks).

**Step 4: Run tests**
Expected: PASS

**Step 5: Commit**

```bash
git commit -m "feat: add RayBackend interface and Kokkos fallback backend"
```

---

### Task 8: Embree backend

**Files:**
- Create: `include/photon/pt/backend/embree_backend.h`
- Create: `src/photon/pt/backend/embree_backend.cpp`
- Modify: `src/photon/pt/backend/backend_factory.cpp`
- Modify: `cmake/Dependencies.cmake`
- Test: `tests/embree_backend_test.cpp` (conditional on Embree availability)
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test**

```cpp
// tests/embree_backend_test.cpp
#ifdef PHOTON_HAS_EMBREE
#include "photon/pt/backend/embree_backend.h"
#include "photon/pt/scene/builder.h"
#include <Kokkos_Core.hpp>
#include <cassert>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;
    auto scene = SceneBuilder::make_two_quads();

    EmbreeBackend backend;
    assert(std::strcmp(backend.name(), "embree") == 0);
    backend.build_accel(scene);

    // Same ray test as Kokkos backend
    RayBatch rays;
    rays.count = 1;
    rays.origins = Kokkos::View<Vec3*>("org", 1);
    rays.directions = Kokkos::View<Vec3*>("dir", 1);
    rays.tmin = Kokkos::View<f32*>("tmin", 1);
    rays.tmax = Kokkos::View<f32*>("tmax", 1);

    auto org_h = Kokkos::create_mirror_view(rays.origins);
    auto dir_h = Kokkos::create_mirror_view(rays.directions);
    auto tmin_h = Kokkos::create_mirror_view(rays.tmin);
    auto tmax_h = Kokkos::create_mirror_view(rays.tmax);
    org_h(0) = {0, 0, 2};
    dir_h(0) = {0, 0, -1};
    tmin_h(0) = 1e-3f;
    tmax_h(0) = 1e30f;
    Kokkos::deep_copy(rays.origins, org_h);
    Kokkos::deep_copy(rays.directions, dir_h);
    Kokkos::deep_copy(rays.tmin, tmin_h);
    Kokkos::deep_copy(rays.tmax, tmax_h);

    HitBatch hits;
    hits.count = 1;
    hits.hits = Kokkos::View<HitResult*>("hits", 1);

    backend.trace_closest(rays, hits);

    auto hits_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hits.hits);
    assert(hits_h(0).hit);
  }
  Kokkos::finalize();
  return 0;
}
#else
int main() { return 0; } // skip if no Embree
#endif
```

**Step 2: Run test to verify it fails**

**Step 3: Implement**

Add to `cmake/Dependencies.cmake`:
```cmake
option(PHOTON_ENABLE_EMBREE "Enable Embree ray tracing backend" ON)
if(PHOTON_ENABLE_EMBREE)
  find_package(embree 4 QUIET)
  if(embree_FOUND)
    target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_EMBREE)
  else()
    FetchContent_Declare(embree
      GIT_REPOSITORY https://github.com/RenderKit/embree.git
      GIT_TAG v4.3.3
    )
    set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
    set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)
    set(EMBREE_TESTING_INTENSITY 0 CACHE STRING "" FORCE)
    FetchContent_MakeAvailable(embree)
    target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_EMBREE)
  endif()
endif()
```

`embree_backend.cpp`:
- Constructor: `rtcNewDevice(nullptr)`
- `build_accel()`: Create Embree scene, add triangle geometries with `rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)`, set vertex/index buffers, commit
- `trace_closest()`: Copy rays from Kokkos views to host, call `rtcIntersect1()` per ray (or `rtcIntersect4/8/16` with SIMD), write results back to Kokkos HitBatch
- `trace_occluded()`: Same but `rtcOccluded1()`
- Destructor: `rtcReleaseScene`, `rtcReleaseDevice`

Update `backend_factory.cpp`: check for `PHOTON_HAS_EMBREE` → prefer EmbreeBackend for CPU.

**Step 4: Run tests**
Expected: PASS (or skip if Embree not found)

**Step 5: Commit**

```bash
git commit -m "feat: add Embree ray tracing backend"
```

---

### Task 9: OptiX backend

**Files:**
- Create: `include/photon/pt/backend/optix_backend.h`
- Create: `src/photon/pt/backend/optix_backend.cpp`
- Create: `src/photon/pt/backend/optix_kernels.cu` (OptiX programs)
- Modify: `src/photon/pt/backend/backend_factory.cpp`
- Modify: `cmake/Dependencies.cmake`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test** (similar structure to Embree, gated by `PHOTON_HAS_OPTIX`)

**Step 2: Run test to verify it fails**

**Step 3: Implement**

OptiX 7+ integration:
- Create `OptixDeviceContext` from CUDA context
- Build acceleration structures (GAS for triangles, IAS for instances)
- Create minimal pipeline with raygen + closest-hit + miss programs
- Closest-hit program writes hit data (t, normal, prim_id, UV) to per-ray payload
- `trace_closest()`: copy rays to CUDA device memory, launch OptiX pipeline, copy results to Kokkos views
- `trace_occluded()`: same but any-hit program terminates immediately

Key: Share CUDA device context between Kokkos::Cuda and OptiX.

Add to `cmake/Dependencies.cmake`:
```cmake
option(PHOTON_ENABLE_OPTIX "Enable OptiX ray tracing backend" OFF)
if(PHOTON_ENABLE_OPTIX)
  find_package(OptiX 7 REQUIRED) # headers only
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OPTIX)
endif()
```

**Step 4: Run tests** (requires NVIDIA GPU with OptiX)

**Step 5: Commit**

```bash
git commit -m "feat: add OptiX ray tracing backend"
```

---

### Task 10: HIP RT backend

**Files:**
- Create: `include/photon/pt/backend/hiprt_backend.h`
- Create: `src/photon/pt/backend/hiprt_backend.cpp`
- Modify: `src/photon/pt/backend/backend_factory.cpp`
- Modify: `cmake/Dependencies.cmake`
- Modify: `src/CMakeLists.txt`

**Step 1: Write the failing test** (similar structure, gated by `PHOTON_HAS_HIPRT`)

**Step 2: Run test to verify it fails**

**Step 3: Implement**

HIP RT integration:
- Create `hiprtContext` from HIP device
- Build geometry AS: `hiprtBuildGeometry()` with triangle lists
- Build scene AS: `hiprtBuildScene()` for instances
- Trace: `hiprtTrace()` kernel that writes hit results
- Share HIP device context between Kokkos::HIP and HIP RT

Add to `cmake/Dependencies.cmake`:
```cmake
option(PHOTON_ENABLE_HIPRT "Enable HIP RT ray tracing backend" OFF)
if(PHOTON_ENABLE_HIPRT)
  find_package(hiprt REQUIRED)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_HIPRT)
endif()
```

**Step 4: Run tests** (requires AMD GPU with ROCm)

**Step 5: Commit**

```bash
git commit -m "feat: add HIP RT ray tracing backend"
```

---

## Phase 3: Wavefront Path Tracer Integrator

---

### Task 11: Scene structure refactor — multi-mesh, instances, materials

**Files:**
- Modify: `include/photon/pt/scene.h`
- Modify: `include/photon/pt/scene/builder.h`
- Modify: `src/photon/pt/scene/builder.cpp`
- Modify: `src/photon/pt/scene.cpp`
- Test: `tests/scene_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write failing test** — create a Scene with multiple meshes, materials, and lights.

**Step 2: Implement the expanded Scene struct:**

```cpp
struct Instance {
  Mat4 transform;
  Mat4 inv_transform;
  u32 mesh_id;
  u32 material_id;
};

struct Scene {
  // Geometry
  Kokkos::View<TriangleMesh*> meshes; // array of meshes (or single merged mesh)
  u32 mesh_count{0};

  // We store a single merged mesh + BVH for simplicity
  TriangleMesh merged_mesh;
  Bvh bvh; // kept for Kokkos fallback; backends build their own

  // Materials
  Kokkos::View<Material*> materials;
  u32 material_count{0};

  // Textures
  TextureAtlas textures;

  // Lights
  Kokkos::View<Light*> lights;
  u32 light_count{0};
  std::optional<EnvironmentMap> env_map;

  // Instances
  Kokkos::View<Instance*> instances;
  u32 instance_count{0};
};
```

Update `SceneBuilder::make_two_quads()` to populate materials.
Add `SceneBuilder::make_cornell_box()` for a richer test scene.

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: expand Scene with multi-mesh, materials, lights, instances"
```

---

### Task 12: Wavefront path tracer — core integrator rewrite

**Files:**
- Rewrite: `include/photon/pt/pathtracer.h`
- Rewrite: `src/photon/pt/pathtracer.cpp`
- Test: `tests/pathtracer_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write failing test**

```cpp
// tests/pathtracer_test.cpp
#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"
#include "photon/pt/backend/kokkos_backend.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    auto scene = SceneBuilder::make_cornell_box();
    auto backend = std::make_unique<KokkosBackend>();
    backend->build_accel(scene);

    PathTracer pt;
    pt.params.width = 64;
    pt.params.height = 64;
    pt.params.samples_per_pixel = 4;
    pt.params.max_depth = 5;
    pt.set_scene(scene);
    pt.set_backend(std::move(backend));

    auto result = pt.render();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);

    // Check that pixels are non-zero (not black) and finite
    int nonzero = 0;
    for (u32 y = 0; y < 64; ++y)
      for (u32 x = 0; x < 64; ++x) {
        auto c = host(y, x);
        assert(c.x == c.x); // not NaN
        if (c.x > 0.f || c.y > 0.f || c.z > 0.f) nonzero++;
      }
    assert(nonzero > 64 * 64 / 2); // most pixels non-black

    // Check AOVs exist
    assert(result.depth.extent(0) == 64);
    assert(result.normal.extent(0) == 64);
    assert(result.albedo.extent(0) == 64);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Implement the wavefront integrator**

New `PathTracer`:
```cpp
struct RenderResult {
  Kokkos::View<Vec3**, Kokkos::LayoutRight> color;
  Kokkos::View<f32**, Kokkos::LayoutRight>  depth;
  Kokkos::View<Vec3**, Kokkos::LayoutRight> normal;
  Kokkos::View<Vec3**, Kokkos::LayoutRight> albedo;
};

struct PathTracer {
  RenderParams params;

  void set_scene(const Scene& scene);
  void set_backend(std::unique_ptr<RayBackend> backend);

  RenderResult render() const;

private:
  Scene m_scene;
  std::unique_ptr<RayBackend> m_backend;
};
```

Wavefront render loop:
```
1. Generate camera rays (all pixels × SPP)
2. For each bounce:
   a. backend->trace_closest(rays, hits)
   b. Kokkos::parallel_for over active rays:
      - If miss: add env map contribution, deactivate ray
      - If hit:
        * Look up material
        * NEE: sample light → generate shadow ray
        * BSDF: sample new direction
        * MIS weight for NEE
        * Russian roulette after bounce 3
        * Write new ray origin/direction
   c. backend->trace_occluded(shadow_rays, occluded)
   d. Kokkos::parallel_for: add NEE contribution where unoccluded
3. Accumulate to pixel buffer
4. Write AOVs on first hit
```

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: rewrite path tracer as wavefront integrator with NEE, MIS, Russian roulette"
```

---

### Task 13: Camera system — ANARI camera with thin lens DOF

**Files:**
- Modify: `include/photon/pt/camera.h`
- Test: `tests/camera_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write failing test** — thin lens camera generates rays with DOF jitter.

**Step 2: Implement**

```cpp
struct Camera {
  Vec3 origin;
  Vec3 lower_left;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;          // ONB
  f32  lens_radius{0.f}; // 0 = pinhole
  f32  focus_dist{1.f};

  static KOKKOS_FUNCTION Camera make_perspective(
      const Vec3& look_from, const Vec3& look_at, const Vec3& vup,
      f32 vfov_degrees, f32 aspect,
      f32 aperture = 0.f, f32 focus_dist = 1.f);

  static KOKKOS_FUNCTION Camera make_orthographic(
      const Vec3& look_from, const Vec3& look_at, const Vec3& vup,
      f32 ortho_height, f32 aspect);

  KOKKOS_FUNCTION Ray ray(f32 s, f32 t, Rng& rng) const; // DOF jitter
};
```

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: add thin lens camera with depth of field"
```

---

## Phase 4: Volumetric Rendering

---

### Task 14: Volume grid and delta tracking

**Files:**
- Create: `include/photon/pt/volume.h`
- Create: `src/photon/pt/volume.cpp`
- Modify: `src/photon/pt/pathtracer.cpp` (integrate volume into path tracing loop)
- Test: `tests/volume_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/CMakeLists.txt`

**Step 1: Write failing test**

```cpp
// tests/volume_test.cpp
#include "photon/pt/volume.h"
#include "photon/pt/rng.h"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    // Create 4x4x4 constant density volume
    VolumeGrid vol;
    vol.dimensions = {4, 4, 4};
    vol.density = Kokkos::View<f32***, Kokkos::LayoutRight>("den", 4, 4, 4);
    Kokkos::deep_copy(vol.density, 1.f); // uniform density
    vol.bounds_lo = {-1, -1, -1};
    vol.bounds_hi = {1, 1, 1};
    vol.sigma_s = {1, 1, 1};
    vol.sigma_a = {0.1f, 0.1f, 0.1f};
    vol.max_density = 1.f;
    vol.g = 0.f; // isotropic

    // Ray through volume
    Rng rng(42);
    Ray ray{{-2, 0, 0}, {1, 0, 0}};

    // Delta tracking should return a scatter event inside the volume
    auto event = delta_tracking(vol, ray, 1e-3f, 100.f, rng);
    // With density=1, sigma_t=1.1, most rays should scatter
    if (event.scattered) {
      assert(event.position.x >= -1.f && event.position.x <= 1.f);
      assert(event.sigma_s.x > 0.f);
    }

    // Henyey-Greenstein phase function
    Vec3 wo{1, 0, 0};
    Vec3 wi = sample_hg(wo, 0.f, rng); // isotropic
    assert(std::abs(length(wi) - 1.f) < 0.01f);

    f32 p = pdf_hg(dot(wo, wi), 0.f);
    assert(p > 0.f);
  }
  Kokkos::finalize();
  return 0;
}
```

**Step 2: Implement**

```cpp
struct VolumeGrid {
  Kokkos::View<f32***, Kokkos::LayoutRight> density;
  Vec3 bounds_lo, bounds_hi;
  f32  max_density;
  Vec3 sigma_s;
  Vec3 sigma_a;
  f32  g;
  Vec3 emission{0, 0, 0};
  f32  emission_strength{0.f};
  u32  dimensions[3]{0, 0, 0};

  KOKKOS_FUNCTION f32 sample_density(const Vec3& world_pos) const;
};

struct VolumeEvent {
  Vec3 position;
  Vec3 sigma_s;
  Vec3 sigma_a;
  f32  g;
  bool scattered;
};

KOKKOS_FUNCTION VolumeEvent delta_tracking(
    const VolumeGrid& vol, const Ray& ray, f32 tmin, f32 tmax, Rng& rng);

KOKKOS_FUNCTION Vec3 sample_hg(const Vec3& wo, f32 g, Rng& rng);
KOKKOS_FUNCTION f32  pdf_hg(f32 cos_theta, f32 g);
```

Integrate into pathtracer:
- Before surface intersection, check if ray passes through any volume
- If scatter event occurs before surface hit: apply phase function, continue path
- If no scatter: proceed to surface shading as normal

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: add volumetric rendering with delta tracking and HG phase function"
```

---

## Phase 5: ANARI Device Completion

---

### Task 15: Refactor SceneFromAnari — extract inline code

**Files:**
- Rewrite: `src/photon/anari/SceneFromAnari.cpp`
- Modify: `src/photon/anari/SceneFromAnari.h`
- Modify: `src/photon/anari/PhotonDevice.cpp` (remove inline scene building)
- Test: `tests/pt_smoke_test.cpp` (update)

**Step 1: Move the 200+ lines of scene building from `PhotonDevice::renderFrame()` into `build_scene_from_anari()`**

The function should:
1. Walk ANARI world → surfaces → geometries
2. Parse vertex.position, vertex.normal, vertex.texcoord arrays
3. Handle triangle and quad geometry subtypes
4. Look up material parameters from ANARI material objects
5. Collect lights from ANARI light objects
6. Build the full `Scene` struct
7. Return `std::optional<Scene>`

**Step 2: Simplify `PhotonDevice::renderFrame()`:**

```cpp
void PhotonDevice::renderFrame(ANARIFrame fb) {
  // ... size parsing ...
  auto scene = build_scene_from_anari(world_handle, *this);
  if (!scene)
    scene = SceneBuilder::make_two_quads();

  auto backend = create_best_backend();
  backend->build_accel(*scene);

  PathTracer pt;
  pt.params = parse_render_params(frame_obj);
  pt.set_scene(*scene);
  pt.set_backend(std::move(backend));
  auto result = pt.render();

  // Copy to framebuffer
  copy_result_to_framebuffer(result, m_fb_bytes, m_fb_w, m_fb_h);
}
```

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "refactor: extract scene building from renderFrame into SceneFromAnari"
```

---

### Task 16: Implement ANARI Camera, Material, Light objects

**Files:**
- Modify: `src/photon/anari/PhotonDevice.cpp`
- Modify: `src/photon/anari/PhotonDevice.h`
- Modify: `src/photon/anari/SceneFromAnari.cpp`
- Test: `tests/anari_objects_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write failing test** — create ANARI camera, material, and light via API.

**Step 2: Implement**

Camera: `newCamera("perspective")` / `newCamera("orthographic")`
- Parameters: `position`, `direction`, `up`, `fovy`, `aspect`, `apertureRadius`, `focusDistance`

Material: `newMaterial("physicallyBased")` / `newMaterial("matte")`
- Parameters: `baseColor`, `metallic`, `roughness`, `ior`, `transmission`, etc.
- Map to Disney Principled BSDF parameters

Light: `newLight("point")` / `newLight("directional")` / `newLight("hdri")`
- Parameters: `position`, `direction`, `color`, `intensity`, `radiance` (for hdri: image array)

Group: `newGroup()` — geometry collection
Instance: `newInstance("transform")` — transform + group reference

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: implement ANARI camera, material, light, group, instance objects"
```

---

### Task 17: ANARI Volume and SpatialField objects

**Files:**
- Modify: `src/photon/anari/PhotonDevice.cpp`
- Modify: `src/photon/anari/SceneFromAnari.cpp`
- Test: `tests/anari_volume_test.cpp`

**Step 1: Write failing test** — create ANARI spatial field + volume.

**Step 2: Implement**

SpatialField: `newSpatialField("structuredRegular")`
- Parameters: `data` (3D array), `origin`, `spacing`

Volume: `newVolume("transferFunction1D")`
- Parameters: `field`, `color` (transfer function), `opacity`, `densityScale`

Map to `VolumeGrid` in SceneFromAnari.

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: implement ANARI Volume and SpatialField objects"
```

---

### Task 18: Progressive rendering and AOV frame channels

**Files:**
- Modify: `src/photon/anari/PhotonDevice.cpp`
- Modify: `src/photon/anari/PhotonDevice.h`
- Modify: `include/photon/pt/pathtracer.h`
- Test: `tests/progressive_test.cpp`

**Step 1: Write failing test** — render progressively, map intermediate results.

**Step 2: Implement**

- `FrameState` accumulation buffer in PhotonDevice
- `renderFrame()` adds samples to accumulation buffer
- `frameReady(ANARI_NO_WAIT)` checks if target SPP reached
- `frameBufferMap("color")` returns current averaged result
- `frameBufferMap("depth")` returns depth AOV
- `frameBufferMap("normal")` returns normal AOV
- `frameBufferMap("albedo")` returns albedo AOV
- Renderer parameter `"pixelSamples"` controls target SPP

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: add progressive rendering and AOV frame channels"
```

---

## Phase 6: Post-Processing and Polish

---

### Task 19: Tone mapping

**Files:**
- Create: `include/photon/pt/tone_mapping.h`
- Modify: `src/photon/anari/PhotonDevice.cpp`
- Test: `tests/tonemap_test.cpp`

**Step 1: Write failing test** — verify ACES tone mapping output.

**Step 2: Implement**

```cpp
KOKKOS_FUNCTION Vec3 aces_tonemap(Vec3 c);
KOKKOS_FUNCTION Vec3 reinhard_tonemap(Vec3 c);
KOKKOS_FUNCTION Vec3 exposure_adjust(Vec3 c, f32 exposure);
```

Apply tone mapping in `frameBufferMap("color")` based on renderer parameter `"toneMapper"`.

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: add ACES and Reinhard tone mapping"
```

---

### Task 20: Intel OIDN denoiser integration

**Files:**
- Create: `include/photon/pt/denoiser.h`
- Create: `src/photon/pt/denoiser.cpp`
- Modify: `cmake/Dependencies.cmake`
- Modify: `src/photon/anari/PhotonDevice.cpp`
- Test: `tests/denoiser_test.cpp`

**Step 1: Write failing test** (gated by `PHOTON_HAS_OIDN`)

**Step 2: Implement**

```cmake
option(PHOTON_ENABLE_OIDN "Enable Intel OIDN denoiser" ON)
if(PHOTON_ENABLE_OIDN)
  find_package(OpenImageDenoise 2 QUIET)
  if(OpenImageDenoise_FOUND)
    target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OIDN)
    target_link_libraries(anari_library_photon PUBLIC OpenImageDenoise)
  endif()
endif()
```

```cpp
struct Denoiser {
  void denoise(
    Kokkos::View<Vec3**, Kokkos::LayoutRight>& color,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& albedo,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& normal,
    u32 width, u32 height);
};
```

Apply when renderer parameter `"denoise"` is true.

**Step 3: Run tests**
**Step 4: Commit**

```bash
git commit -m "feat: add Intel OIDN denoiser integration"
```

---

### Task 21: Cornell box test scene and PPM validation app

**Files:**
- Modify: `src/photon/pt/scene/builder.cpp`
- Modify: `include/photon/pt/scene/builder.h`
- Modify: `app/photon_render.cpp`
- Modify: `app/photon_anari_render.cpp`

**Step 1: Implement `SceneBuilder::make_cornell_box()`**

Classic Cornell box with:
- White walls, red left wall, green right wall
- White ceiling light (emissive material)
- Two boxes (one diffuse, one metallic)
- Proper materials using Disney BSDF

**Step 2: Update apps to use new path tracer and backend**

Update `photon_render.cpp` to use `create_best_backend()` and the new `PathTracer`.
Update `photon_anari_render.cpp` similarly.

**Step 3: Build and render test images**

```bash
./build/app/photon_render && open out.ppm
```

Verify Cornell box renders correctly with:
- Colored walls from materials
- Light from ceiling emissive
- Shadows from NEE
- Metallic reflection on one box

**Step 4: Commit**

```bash
git commit -m "feat: add Cornell box test scene and update rendering apps"
```

---

### Task 22: Integration test and smoke tests

**Files:**
- Modify: `tests/pt_smoke_test.cpp`
- Create: `tests/integration_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Write comprehensive integration tests**

```cpp
// tests/integration_test.cpp
// Test full pipeline: Scene → Backend → PathTracer → Result
// Test each backend (Kokkos, Embree if available)
// Test with materials, lights, env map
// Test progressive rendering
// Test AOVs
// Test volume rendering
// Verify energy conservation (white furnace test)
```

**Step 2: Run all tests**

```bash
cmake --build build -j && ctest --test-dir build -R '^photon_'
```

**Step 3: Commit**

```bash
git commit -m "test: add comprehensive integration tests"
```

---

## Phase Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| 1: Foundation | 1-6 | Math, sampling, materials, lights, textures |
| 2: Backends | 7-10 | RayBackend interface, Kokkos/Embree/OptiX/HIPRT |
| 3: Integrator | 11-13 | Wavefront PT with NEE/MIS/RR, camera |
| 4: Volumes | 14 | Delta tracking, phase functions |
| 5: ANARI | 15-18 | Full device completion, progressive rendering |
| 6: Polish | 19-22 | Tone mapping, denoising, test scenes, tests |

Total: 22 tasks, each independently testable and committable.
