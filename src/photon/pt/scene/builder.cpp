#include "photon/pt/scene/builder.h"

namespace photon::pt {

static Material make_diffuse(Vec3 base_color)
{
  Material m{};
  m.base_color = base_color;
  m.metallic = 0.f;
  m.roughness = 1.f;
  return m;
}

static Material make_emissive(Vec3 emission, f32 strength)
{
  Material m{};
  m.base_color = {0.f, 0.f, 0.f};
  m.emission = emission;
  m.emission_strength = strength;
  m.roughness = 1.f;
  return m;
}

static Material make_metal(Vec3 base_color, f32 roughness)
{
  Material m{};
  m.base_color = base_color;
  m.metallic = 1.f;
  m.roughness = roughness;
  return m;
}

Scene SceneBuilder::make_two_quads()
{
  TriangleMesh mesh;

  mesh.positions = Kokkos::View<Vec3 *>("pos", 8);
  mesh.indices = Kokkos::View<u32 *>("idx", 12);
  mesh.albedo_per_prim = Kokkos::View<Vec3 *>("alb", 4);
  mesh.material_ids = Kokkos::View<u32 *>("mat", 4);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);
  auto mat_h = Kokkos::create_mirror_view(mesh.material_ids);

  pos_h(0) = {-1.f, -1.f, -2.5f};
  pos_h(1) = {+1.f, -1.f, -2.5f};
  pos_h(2) = {+1.f, +1.f, -2.5f};
  pos_h(3) = {-1.f, +1.f, -2.5f};

  pos_h(4) = {-2.f, -1.f, -1.5f};
  pos_h(5) = {+2.f, -1.f, -1.5f};
  pos_h(6) = {+2.f, -1.f, -5.5f};
  pos_h(7) = {-2.f, -1.f, -5.5f};

  idx_h(0) = 0;
  idx_h(1) = 1;
  idx_h(2) = 2;
  idx_h(3) = 0;
  idx_h(4) = 2;
  idx_h(5) = 3;

  idx_h(6) = 4;
  idx_h(7) = 5;
  idx_h(8) = 6;
  idx_h(9) = 4;
  idx_h(10) = 6;
  idx_h(11) = 7;

  alb_h(0) = {0.9f, 0.2f, 0.2f};
  alb_h(1) = {0.2f, 0.9f, 0.2f};
  alb_h(2) = {0.7f, 0.7f, 0.7f};
  alb_h(3) = {0.7f, 0.7f, 0.7f};

  mat_h(0) = 0;
  mat_h(1) = 0;
  mat_h(2) = 0;
  mat_h(3) = 0;

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);
  Kokkos::deep_copy(mesh.material_ids, mat_h);

  Scene s;
  s.mesh = mesh;
  s.bvh = Bvh::build_cpu(mesh);

  s.materials = Kokkos::View<Material *>("materials", 1);
  s.material_count = 1;
  auto mats_h = Kokkos::create_mirror_view(s.materials);
  mats_h(0) = make_diffuse({0.7f, 0.7f, 0.7f});
  Kokkos::deep_copy(s.materials, mats_h);

  s.lights = Kokkos::View<Light *>("lights", 0);
  s.light_count = 0;

  s.emissive_prim_ids = Kokkos::View<u32 *>("emissive_ids", 0);
  s.emissive_prim_areas = Kokkos::View<f32 *>("emissive_areas", 0);
  s.emissive_count = 0;
  s.total_emissive_area = 0.f;

  return s;
}

struct PrimBuilder {
  decltype(Kokkos::create_mirror_view(Kokkos::View<Vec3 *>("", 0))) &pos;
  decltype(Kokkos::create_mirror_view(Kokkos::View<u32 *>("", 0))) &idx;
  decltype(Kokkos::create_mirror_view(Kokkos::View<u32 *>("", 0))) &mat;
  u32 pos_cursor{0};
  u32 tri_cursor{0};

  void add_quad(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d, u32 mat_id)
  {
    const u32 base = pos_cursor;
    pos(base + 0) = a;
    pos(base + 1) = b;
    pos(base + 2) = c;
    pos(base + 3) = d;

    const u32 i = tri_cursor * 3;
    idx(i + 0) = base + 0;
    idx(i + 1) = base + 1;
    idx(i + 2) = base + 2;
    mat(tri_cursor + 0) = mat_id;
    tri_cursor++;

    const u32 j = tri_cursor * 3;
    idx(j + 0) = base + 0;
    idx(j + 1) = base + 2;
    idx(j + 2) = base + 3;
    mat(tri_cursor + 0) = mat_id;
    tri_cursor++;

    pos_cursor += 4;
  }
};

Scene SceneBuilder::make_cornell_box()
{
  constexpr f32 s = 1.f;

  TriangleMesh mesh;
  mesh.positions = Kokkos::View<Vec3 *>("pos", 72);
  mesh.indices = Kokkos::View<u32 *>("idx", 108);
  mesh.material_ids = Kokkos::View<u32 *>("mat", 36);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto mat_h = Kokkos::create_mirror_view(mesh.material_ids);

  PrimBuilder prim{pos_h, idx_h, mat_h};

  const u32 MAT_WHITE = 0;
  const u32 MAT_RED = 1;
  const u32 MAT_GREEN = 2;
  const u32 MAT_LIGHT = 3;
  const u32 MAT_TALL_METAL = 4;
  const u32 MAT_SHORT_WHITE = 5;

  prim.add_quad({-s, -s, -s}, {+s, -s, -s}, {+s, -s, +s}, {-s, -s, +s}, MAT_WHITE);
  prim.add_quad({-s, +s, +s}, {+s, +s, +s}, {+s, +s, -s}, {-s, +s, -s}, MAT_WHITE);
  prim.add_quad({-s, -s, +s}, {+s, -s, +s}, {+s, +s, +s}, {-s, +s, +s}, MAT_WHITE);

  prim.add_quad({-s, -s, -s}, {-s, -s, +s}, {-s, +s, +s}, {-s, +s, -s}, MAT_RED);
  prim.add_quad({+s, -s, +s}, {+s, -s, -s}, {+s, +s, -s}, {+s, +s, +s}, MAT_GREEN);

  const f32 lx0 = -0.25f;
  const f32 lx1 = 0.25f;
  const f32 lz0 = -0.25f;
  const f32 lz1 = 0.25f;
  const f32 ly = +s - 1e-3f;
  prim.add_quad({lx0, ly, lz0}, {lx1, ly, lz0}, {lx1, ly, lz1}, {lx0, ly, lz1}, MAT_LIGHT);

  const f32 b0 = -0.35f;
  const f32 b1 = 0.0f;
  const f32 b2 = -0.2f;
  const f32 b3 = 0.2f;
  const f32 h0 = -s;
  const f32 h1 = +0.6f;

  prim.add_quad({b0, h0, b2}, {b1, h0, b2}, {b1, h0, b3}, {b0, h0, b3}, MAT_TALL_METAL);
  prim.add_quad({b0, h1, b2}, {b1, h1, b2}, {b1, h1, b3}, {b0, h1, b3}, MAT_TALL_METAL);
  prim.add_quad({b0, h0, b2}, {b0, h0, b3}, {b0, h1, b3}, {b0, h1, b2}, MAT_TALL_METAL);
  prim.add_quad({b1, h0, b3}, {b1, h0, b2}, {b1, h1, b2}, {b1, h1, b3}, MAT_TALL_METAL);
  prim.add_quad({b0, h0, b3}, {b1, h0, b3}, {b1, h1, b3}, {b0, h1, b3}, MAT_TALL_METAL);
  prim.add_quad({b1, h0, b2}, {b0, h0, b2}, {b0, h1, b2}, {b1, h1, b2}, MAT_TALL_METAL);

  const f32 c0 = 0.15f;
  const f32 c1 = 0.55f;
  const f32 c2 = -0.1f;
  const f32 c3 = 0.3f;
  const f32 ch0 = -s;
  const f32 ch1 = -0.2f;

  prim.add_quad({c0, ch0, c2}, {c1, ch0, c2}, {c1, ch0, c3}, {c0, ch0, c3}, MAT_SHORT_WHITE);
  prim.add_quad({c0, ch1, c2}, {c1, ch1, c2}, {c1, ch1, c3}, {c0, ch1, c3}, MAT_SHORT_WHITE);
  prim.add_quad({c0, ch0, c2}, {c0, ch0, c3}, {c0, ch1, c3}, {c0, ch1, c2}, MAT_SHORT_WHITE);
  prim.add_quad({c1, ch0, c3}, {c1, ch0, c2}, {c1, ch1, c2}, {c1, ch1, c3}, MAT_SHORT_WHITE);
  prim.add_quad({c0, ch0, c3}, {c1, ch0, c3}, {c1, ch1, c3}, {c0, ch1, c3}, MAT_SHORT_WHITE);
  prim.add_quad({c1, ch0, c2}, {c0, ch0, c2}, {c0, ch1, c2}, {c1, ch1, c2}, MAT_SHORT_WHITE);

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.material_ids, mat_h);

  mesh.albedo_per_prim = Kokkos::View<Vec3 *>("alb", mesh.triangle_count());
  auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);
  for (u32 tri = 0; tri < mesh.triangle_count(); ++tri)
    alb_h(tri) = {0.73f, 0.73f, 0.73f};
  Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);

  Scene scene;
  scene.mesh = mesh;
  scene.bvh = Bvh::build_cpu(mesh);

  scene.materials = Kokkos::View<Material *>("materials", 6);
  scene.material_count = 6;

  auto mats_h = Kokkos::create_mirror_view(scene.materials);
  mats_h(MAT_WHITE) = make_diffuse({0.73f, 0.73f, 0.73f});
  mats_h(MAT_RED) = make_diffuse({0.65f, 0.05f, 0.05f});
  mats_h(MAT_GREEN) = make_diffuse({0.12f, 0.45f, 0.15f});
  mats_h(MAT_LIGHT) = make_emissive({15.f, 15.f, 15.f}, 1.f);
  mats_h(MAT_TALL_METAL) = make_metal({0.73f, 0.73f, 0.73f}, 0.05f);
  mats_h(MAT_SHORT_WHITE) = make_diffuse({0.73f, 0.73f, 0.73f});
  Kokkos::deep_copy(scene.materials, mats_h);

  scene.emissive_count = 2;
  scene.emissive_prim_ids = Kokkos::View<u32 *>("emissive_ids", scene.emissive_count);
  scene.emissive_prim_areas = Kokkos::View<f32 *>("emissive_areas", scene.emissive_count);

  auto eids_h = Kokkos::create_mirror_view(scene.emissive_prim_ids);
  auto eareas_h = Kokkos::create_mirror_view(scene.emissive_prim_areas);

  eids_h(0) = 10;
  eids_h(1) = 11;

  const Vec3 la = pos_h(20 + 0);
  const Vec3 lb = pos_h(20 + 1);
  const Vec3 lc = pos_h(20 + 2);
  const Vec3 ld = pos_h(20 + 3);

  const f32 area = 0.5f * length(cross(lb - la, lc - la)) + 0.5f * length(cross(lc - la, ld - la));
  eareas_h(0) = area * 0.5f;
  eareas_h(1) = area * 0.5f;

  scene.total_emissive_area = area;

  Kokkos::deep_copy(scene.emissive_prim_ids, eids_h);
  Kokkos::deep_copy(scene.emissive_prim_areas, eareas_h);

  scene.lights = Kokkos::View<Light *>("lights", 1);
  scene.light_count = 1;
  auto lights_h = Kokkos::create_mirror_view(scene.lights);

  Light l{};
  l.type = LightType::Area;
  l.mesh_prim_begin = 10;
  l.mesh_prim_count = 2;
  l.area = area;
  l.color = {15.f, 15.f, 15.f};
  l.intensity = 1.f;
  lights_h(0) = l;
  Kokkos::deep_copy(scene.lights, lights_h);

  return scene;
}

} // namespace photon::pt
