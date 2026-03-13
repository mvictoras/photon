#include "photon/anari/SceneFromAnari.h"

#include <cmath>
#include <cstring>

#include <Kokkos_Core.hpp>

#include "photon/anari/PhotonDevice.h"

#include <optional>
#include <vector>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/camera.h"
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/math.h"
#include "photon/pt/math_mat4.h"
#include "photon/pt/scene.h"

namespace photon::anari_device {

namespace {

constexpr float PI = 3.1415926535f;

template <typename T>
bool read_param(const PhotonDevice::Object *obj, const char *name, T &out)
{
  if (!obj)
    return false;
  const auto it = obj->params.find(name);
  if (it == obj->params.end() || it->second.size() != sizeof(T))
    return false;
  std::memcpy(&out, it->second.data(), sizeof(T));
  return true;
}

bool read_handle(const PhotonDevice::Object *obj, const char *name, uintptr_t &out)
{
  return read_param(obj, name, out);
}

const char *read_string_param(const PhotonDevice::Object *obj, const char *name)
{
  if (!obj)
    return nullptr;
  const auto it = obj->params.find(name);
  if (it == obj->params.end() || it->second.empty())
    return nullptr;
  return reinterpret_cast<const char *>(it->second.data());
}

const PhotonDevice::Object *read_handle_object(
    const PhotonDevice &dev, const PhotonDevice::Object *obj, const char *name, ANARIDataType expectedType)
{
  uintptr_t h = 0;
  if (!read_handle(obj, name, h) || h == 0)
    return nullptr;
  const auto *o = dev.getObject(h);
  if (!o || o->object_type != expectedType)
    return nullptr;
  return o;
}

const PhotonDevice::Object *read_array1d(
    const PhotonDevice &dev, const PhotonDevice::Object *obj, const char *name, ANARIDataType elementType)
{
  auto *a = read_handle_object(dev, obj, name, ANARI_ARRAY1D);
  if (!a || !a->memory || a->array_num_items1 == 0)
    return nullptr;
  if (elementType != ANARI_UNKNOWN && a->array_element_type != elementType)
    return nullptr;
  return a;
}

photon::pt::Vec3 v3_from_floats(const float *p)
{
  return photon::pt::Vec3{p[0], p[1], p[2]};
}

photon::pt::Vec3 v3_from_vec4(const float *p)
{
  return photon::pt::Vec3{p[0], p[1], p[2]};
}

struct ImportedMaterial
{
  photon::pt::Material mat;
  bool use_vertex_color{false};
};

ImportedMaterial import_material(const PhotonDevice &dev, const PhotonDevice::Object *surface)
{
  ImportedMaterial out;

  uintptr_t mh = 0;
  if (!read_handle(surface, "material", mh) || mh == 0)
    return out;

  const auto *mo = dev.getObject(mh);
  if (!mo || mo->object_type != ANARI_MATERIAL)
    return out;

  const char *subtype = read_string_param(mo, "subtype");
  if (!subtype)
    return out;

  if (std::strcmp(subtype, "matte") == 0) {
    photon::pt::Vec3 c{0.8f, 0.8f, 0.8f};
    if (read_param(mo, "color", c)) {
      out.mat.base_color = c;
    } else {
      const char *colorStr = read_string_param(mo, "color");
      if (colorStr && std::strcmp(colorStr, "color") == 0)
        out.use_vertex_color = true;
    }
    return out;
  }

  if (std::strcmp(subtype, "physicallyBased") == 0) {
    photon::pt::Vec3 base{0.8f, 0.8f, 0.8f};
    read_param(mo, "baseColor", base);
    out.mat.base_color = base;

    read_param(mo, "metallic", out.mat.metallic);
    read_param(mo, "roughness", out.mat.roughness);
    read_param(mo, "ior", out.mat.ior);
    read_param(mo, "transmission", out.mat.transmission);
    read_param(mo, "specular", out.mat.specular);
    return out;
  }

  return out;
}

struct TriIdx
{
  uint32_t a;
  uint32_t b;
  uint32_t c;
};

// Intermediate per-surface geometry data (host-side, before merging into TriangleMesh)
struct SurfaceGeometry
{
  std::vector<photon::pt::Vec3> positions; // triangle soup: 3 verts per tri
  std::vector<TriIdx> triangles;           // indices into positions
  ImportedMaterial material;
  std::vector<photon::pt::Vec3> vertex_colors; // if use_vertex_color, one per vertex in positions
};

// ─── Sphere tessellation ────────────────────────────────────────────────────

void tessellate_sphere(const photon::pt::Vec3 &center, float radius,
    uint32_t stacks, uint32_t slices,
    std::vector<photon::pt::Vec3> &out_pos,
    std::vector<TriIdx> &out_tri)
{
  const uint32_t base_vert = uint32_t(out_pos.size());

  // Generate vertices
  // Top pole
  out_pos.push_back(center + photon::pt::Vec3{0.f, radius, 0.f});

  for (uint32_t i = 1; i < stacks; ++i) {
    const float phi = PI * float(i) / float(stacks);
    const float sp = std::sin(phi);
    const float cp = std::cos(phi);
    for (uint32_t j = 0; j < slices; ++j) {
      const float theta = 2.f * PI * float(j) / float(slices);
      const float st = std::sin(theta);
      const float ct = std::cos(theta);
      out_pos.push_back(center + photon::pt::Vec3{radius * sp * ct, radius * cp, radius * sp * st});
    }
  }

  // Bottom pole
  out_pos.push_back(center + photon::pt::Vec3{0.f, -radius, 0.f});

  const uint32_t top = base_vert;
  const uint32_t bottom = uint32_t(out_pos.size()) - 1;

  // Top cap triangles
  for (uint32_t j = 0; j < slices; ++j) {
    const uint32_t j1 = (j + 1) % slices;
    out_tri.push_back({top, base_vert + 1 + j, base_vert + 1 + j1});
  }

  // Middle rows
  for (uint32_t i = 0; i < stacks - 2; ++i) {
    for (uint32_t j = 0; j < slices; ++j) {
      const uint32_t j1 = (j + 1) % slices;
      const uint32_t cur = base_vert + 1 + i * slices + j;
      const uint32_t nxt = base_vert + 1 + i * slices + j1;
      const uint32_t cur_below = base_vert + 1 + (i + 1) * slices + j;
      const uint32_t nxt_below = base_vert + 1 + (i + 1) * slices + j1;
      out_tri.push_back({cur, cur_below, nxt});
      out_tri.push_back({nxt, cur_below, nxt_below});
    }
  }

  // Bottom cap triangles
  const uint32_t last_ring = base_vert + 1 + (stacks - 2) * slices;
  for (uint32_t j = 0; j < slices; ++j) {
    const uint32_t j1 = (j + 1) % slices;
    out_tri.push_back({last_ring + j, bottom, last_ring + j1});
  }
}

// ─── Cylinder tessellation ──────────────────────────────────────────────────

void tessellate_cylinder(const photon::pt::Vec3 &p0, const photon::pt::Vec3 &p1,
    float radius, uint32_t segments,
    std::vector<photon::pt::Vec3> &out_pos,
    std::vector<TriIdx> &out_tri)
{
  using namespace photon::pt;
  const uint32_t base_vert = uint32_t(out_pos.size());

  const Vec3 axis = p1 - p0;
  const float h = length(axis);
  if (h < 1e-10f)
    return;

  const Vec3 up = axis * (1.f / h);

  // Build orthonormal basis around the axis
  Vec3 perp;
  if (std::fabs(up.x) < 0.9f)
    perp = normalize(cross(up, Vec3{1.f, 0.f, 0.f}));
  else
    perp = normalize(cross(up, Vec3{0.f, 1.f, 0.f}));
  const Vec3 bi = cross(up, perp);

  // Generate ring vertices at p0 and p1
  for (uint32_t j = 0; j < segments; ++j) {
    const float theta = 2.f * PI * float(j) / float(segments);
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    const Vec3 offset = perp * (radius * ct) + bi * (radius * st);
    out_pos.push_back(p0 + offset); // bottom ring
  }
  for (uint32_t j = 0; j < segments; ++j) {
    const float theta = 2.f * PI * float(j) / float(segments);
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    const Vec3 offset = perp * (radius * ct) + bi * (radius * st);
    out_pos.push_back(p1 + offset); // top ring
  }

  // Side triangles
  for (uint32_t j = 0; j < segments; ++j) {
    const uint32_t j1 = (j + 1) % segments;
    const uint32_t b0 = base_vert + j;
    const uint32_t b1 = base_vert + j1;
    const uint32_t t0 = base_vert + segments + j;
    const uint32_t t1 = base_vert + segments + j1;
    out_tri.push_back({b0, b1, t0});
    out_tri.push_back({t0, b1, t1});
  }

  // Bottom cap (fan from center)
  const uint32_t bot_center = uint32_t(out_pos.size());
  out_pos.push_back(p0);
  for (uint32_t j = 0; j < segments; ++j) {
    const uint32_t j1 = (j + 1) % segments;
    out_tri.push_back({bot_center, base_vert + j1, base_vert + j});
  }

  // Top cap (fan from center)
  const uint32_t top_center = uint32_t(out_pos.size());
  out_pos.push_back(p1);
  for (uint32_t j = 0; j < segments; ++j) {
    const uint32_t j1 = (j + 1) % segments;
    out_tri.push_back({top_center, base_vert + segments + j, base_vert + segments + j1});
  }
}

// ─── Process a single surface into SurfaceGeometry ──────────────────────────

bool process_surface(const PhotonDevice &dev, const PhotonDevice::Object *so, SurfaceGeometry &sg)
{
  using namespace photon::pt;

  uintptr_t geom_h = 0;
  if (!read_handle(so, "geometry", geom_h) || geom_h == 0)
    return false;
  auto *go = dev.getObject(geom_h);
  if (!go || go->object_type != ANARI_GEOMETRY)
    return false;

  sg.material = import_material(dev, so);

  const char *subtype = read_string_param(go, "subtype");

  // ── Sphere geometry ──
  if (subtype && std::strcmp(subtype, "sphere") == 0) {
    const auto *posA = read_array1d(dev, go, "vertex.position", ANARI_FLOAT32_VEC3);
    if (!posA)
      return false;

    const float *pos = reinterpret_cast<const float *>(posA->memory);
    const uint64_t nspheres = posA->array_num_items1;

    // Per-vertex radius
    const auto *radA = read_array1d(dev, go, "vertex.radius", ANARI_FLOAT32);
    const float *per_radius = radA ? reinterpret_cast<const float *>(radA->memory) : nullptr;

    // Uniform radius
    float uniform_radius = 0.01f;
    read_param(go, "radius", uniform_radius);

    // Per-vertex color
    const auto *colA = read_array1d(dev, go, "vertex.color", ANARI_FLOAT32_VEC4);
    const float *col = colA ? reinterpret_cast<const float *>(colA->memory) : nullptr;

    constexpr uint32_t stacks = 8;
    constexpr uint32_t slices = 16;

    for (uint64_t si = 0; si < nspheres; ++si) {
      const Vec3 center = v3_from_floats(pos + 3 * si);
      const float r = per_radius ? per_radius[si] : uniform_radius;

      const size_t pre_verts = sg.positions.size();
      tessellate_sphere(center, r, stacks, slices, sg.positions, sg.triangles);

      if (sg.material.use_vertex_color && col) {
        const Vec3 c = v3_from_vec4(col + 4 * si);
        const size_t new_verts = sg.positions.size() - pre_verts;
        for (size_t v = 0; v < new_verts; ++v)
          sg.vertex_colors.push_back(c);
      }
    }

    return !sg.triangles.empty();
  }

  // ── Cylinder geometry ──
  if (subtype && std::strcmp(subtype, "cylinder") == 0) {
    const auto *posA = read_array1d(dev, go, "vertex.position", ANARI_FLOAT32_VEC3);
    if (!posA)
      return false;

    const float *pos = reinterpret_cast<const float *>(posA->memory);
    const uint64_t nv = posA->array_num_items1;

    // Radius
    const auto *radA = read_array1d(dev, go, "vertex.radius", ANARI_FLOAT32);
    const float *per_radius = radA ? reinterpret_cast<const float *>(radA->memory) : nullptr;
    float uniform_radius = 0.01f;
    read_param(go, "radius", uniform_radius);

    // Index array (UINT32_VEC2 — pairs of indices into vertex.position)
    uintptr_t idx_h = 0;
    read_handle(go, "primitive.index", idx_h);
    const auto *idxA = idx_h ? dev.getObject(idx_h) : nullptr;

    constexpr uint32_t segments = 12;

    if (idxA && idxA->object_type == ANARI_ARRAY1D && idxA->memory && idxA->array_element_type == ANARI_UINT32_VEC2) {
      const uint32_t *idx = reinterpret_cast<const uint32_t *>(idxA->memory);
      for (uint64_t ci = 0; ci < idxA->array_num_items1; ++ci) {
        const uint32_t i0 = idx[2 * ci + 0];
        const uint32_t i1 = idx[2 * ci + 1];
        const Vec3 p0 = v3_from_floats(pos + 3 * i0);
        const Vec3 p1 = v3_from_floats(pos + 3 * i1);
        const float r = per_radius ? per_radius[ci] : uniform_radius;
        tessellate_cylinder(p0, p1, r, segments, sg.positions, sg.triangles);
      }
    } else {
      // Alternating pairs: vertex[2*i] and vertex[2*i+1]
      const uint64_t ncyl = nv / 2;
      for (uint64_t ci = 0; ci < ncyl; ++ci) {
        const Vec3 p0 = v3_from_floats(pos + 3 * (2 * ci));
        const Vec3 p1 = v3_from_floats(pos + 3 * (2 * ci + 1));
        const float r = per_radius ? per_radius[ci] : uniform_radius;
        tessellate_cylinder(p0, p1, r, segments, sg.positions, sg.triangles);
      }
    }

    return !sg.triangles.empty();
  }

  // ── Triangle / quad geometry (original path) ──
  const auto *posA = read_array1d(dev, go, "vertex.position", ANARI_FLOAT32_VEC3);
  if (!posA)
    return false;

  const float *pos = reinterpret_cast<const float *>(posA->memory);

  const auto *colA = read_array1d(dev, go, "vertex.color", ANARI_FLOAT32_VEC4);
  const float *col = colA ? reinterpret_cast<const float *>(colA->memory) : nullptr;

  // Copy all vertex positions into positions array
  const uint64_t nv = posA->array_num_items1;
  const uint32_t base_vert = uint32_t(sg.positions.size());
  for (uint64_t vi = 0; vi < nv; ++vi)
    sg.positions.push_back(v3_from_floats(pos + 3 * vi));

  // Copy vertex colors if applicable
  if (sg.material.use_vertex_color && col) {
    for (uint64_t vi = 0; vi < nv; ++vi)
      sg.vertex_colors.push_back(v3_from_vec4(col + 4 * vi));
  }

  // Build triangle indices
  uintptr_t idxHandle = 0;
  read_handle(go, "primitive.index", idxHandle);
  const auto *idxA = idxHandle ? dev.getObject(idxHandle) : nullptr;

  if (idxA && idxA->object_type == ANARI_ARRAY1D && idxA->memory) {
    if (idxA->array_element_type == ANARI_UINT32_VEC3) {
      const uint32_t *src = reinterpret_cast<const uint32_t *>(idxA->memory);
      for (uint64_t i = 0; i < idxA->array_num_items1; ++i)
        sg.triangles.push_back({base_vert + src[3 * i], base_vert + src[3 * i + 1], base_vert + src[3 * i + 2]});
    } else if (idxA->array_element_type == ANARI_UINT32_VEC4) {
      const uint32_t *src = reinterpret_cast<const uint32_t *>(idxA->memory);
      for (uint64_t i = 0; i < idxA->array_num_items1; ++i) {
        const uint32_t a = base_vert + src[4 * i];
        const uint32_t b = base_vert + src[4 * i + 1];
        const uint32_t c = base_vert + src[4 * i + 2];
        const uint32_t d = base_vert + src[4 * i + 3];
        sg.triangles.push_back({a, b, c});
        sg.triangles.push_back({a, c, d});
      }
    } else {
      return false;
    }
  } else {
    if (subtype && std::strcmp(subtype, "quad") == 0) {
      const uint64_t nq = nv / 4;
      for (uint64_t q = 0; q < nq; ++q) {
        const uint32_t b = base_vert + uint32_t(4 * q);
        sg.triangles.push_back({b, b + 1, b + 2});
        sg.triangles.push_back({b, b + 2, b + 3});
      }
    } else {
      const uint64_t nt = nv / 3;
      for (uint64_t t = 0; t < nt; ++t) {
        const uint32_t b = base_vert + uint32_t(3 * t);
        sg.triangles.push_back({b, b + 1, b + 2});
      }
    }
  }

  return !sg.triangles.empty();
}

// ─── Collect surfaces from a surface handle array ───────────────────────────

void collect_surfaces_from_array(const PhotonDevice &dev,
    const uintptr_t *surface_handles, uint64_t count,
    std::vector<SurfaceGeometry> &surfaces)
{
  for (uint64_t si = 0; si < count; ++si) {
    auto *so = dev.getObject(surface_handles[si]);
    if (!so || so->object_type != ANARI_SURFACE)
      continue;

    SurfaceGeometry sg;
    if (process_surface(dev, so, sg))
      surfaces.push_back(std::move(sg));
  }
}

// ─── Apply a Mat4 transform to all positions in a SurfaceGeometry ───────────

void apply_transform(SurfaceGeometry &sg, const photon::pt::Mat4 &xfm)
{
  for (auto &p : sg.positions)
    p = xfm.transform_point(p);
}

// ─── Read column-major float[16] into row-major Mat4 ────────────────────────
// ANARI uses column-major matrices (OpenGL style): m[col][row]
// Our Mat4 is row-major: m[row][col]

photon::pt::Mat4 mat4_from_column_major(const float *cm)
{
  photon::pt::Mat4 m{};
  for (int col = 0; col < 4; ++col)
    for (int row = 0; row < 4; ++row)
      m.m[row][col] = cm[col * 4 + row];
  return m;
}

// ─── Build the final merged scene ───────────────────────────────────────────

void merge_surfaces_to_mesh(const std::vector<SurfaceGeometry> &surfaces,
    photon::pt::TriangleMesh &mesh,
    std::vector<photon::pt::Material> &matsCpu)
{
  // Count totals
  uint64_t total_tris = 0;
  for (const auto &sg : surfaces)
    total_tris += sg.triangles.size();

  if (total_tris == 0)
    return;

  const uint64_t total_verts = total_tris * 3;

  mesh.positions = Kokkos::View<photon::pt::Vec3 *>("pos", total_verts);
  mesh.indices = Kokkos::View<photon::pt::u32 *>("idx", total_verts);
  mesh.material_ids = Kokkos::View<photon::pt::u32 *>("mat_id", total_tris);
  mesh.albedo_per_prim = Kokkos::View<photon::pt::Vec3 *>("alb", total_tris);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto mat_id_h = Kokkos::create_mirror_view(mesh.material_ids);
  auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);

  uint64_t vert_off = 0;
  uint64_t tri_off = 0;

  for (const auto &sg : surfaces) {
    const bool use_vcol = sg.material.use_vertex_color && !sg.vertex_colors.empty();

    if (use_vcol) {
      // Per-triangle material from averaged vertex colors
      for (const auto &t : sg.triangles) {
        const auto &p0 = sg.positions[t.a];
        const auto &p1 = sg.positions[t.b];
        const auto &p2 = sg.positions[t.c];

        pos_h(vert_off + 0) = p0;
        pos_h(vert_off + 1) = p1;
        pos_h(vert_off + 2) = p2;

        idx_h(vert_off + 0) = photon::pt::u32(vert_off + 0);
        idx_h(vert_off + 1) = photon::pt::u32(vert_off + 1);
        idx_h(vert_off + 2) = photon::pt::u32(vert_off + 2);

        // Get vertex colors (clamped to available range)
        const auto &c0 = (t.a < sg.vertex_colors.size()) ? sg.vertex_colors[t.a] : sg.material.mat.base_color;
        const auto &c1 = (t.b < sg.vertex_colors.size()) ? sg.vertex_colors[t.b] : sg.material.mat.base_color;
        const auto &c2 = (t.c < sg.vertex_colors.size()) ? sg.vertex_colors[t.c] : sg.material.mat.base_color;
        const photon::pt::Vec3 alb = (c0 + c1 + c2) * (1.f / 3.f);

        const uint32_t matId = uint32_t(matsCpu.size());
        photon::pt::Material triMat = sg.material.mat;
        triMat.base_color = alb;
        matsCpu.push_back(triMat);

        mat_id_h(tri_off) = matId;
        alb_h(tri_off) = alb;

        vert_off += 3;
        tri_off += 1;
      }
    } else {
      // Shared material for all triangles in this surface
      const uint32_t matId = uint32_t(matsCpu.size());
      matsCpu.push_back(sg.material.mat);

      for (const auto &t : sg.triangles) {
        pos_h(vert_off + 0) = sg.positions[t.a];
        pos_h(vert_off + 1) = sg.positions[t.b];
        pos_h(vert_off + 2) = sg.positions[t.c];

        idx_h(vert_off + 0) = photon::pt::u32(vert_off + 0);
        idx_h(vert_off + 1) = photon::pt::u32(vert_off + 1);
        idx_h(vert_off + 2) = photon::pt::u32(vert_off + 2);

        mat_id_h(tri_off) = matId;
        alb_h(tri_off) = sg.material.mat.base_color;

        vert_off += 3;
        tri_off += 1;
      }
    }
  }

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.material_ids, mat_id_h);
  Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);
}

} // anonymous namespace

std::optional<photon::pt::Scene> build_scene_from_anari(ANARIWorld world, const PhotonDevice &dev)
{
  if (!world)
    return std::nullopt;

  auto *wo = dev.getObject(uintptr_t(world));
  if (!wo)
    return std::nullopt;

  std::vector<SurfaceGeometry> all_surfaces;

  // ── Collect direct surfaces from world.surface array ──
  uintptr_t surf_arr_h = 0;
  if (read_handle(wo, "surface", surf_arr_h) && surf_arr_h != 0) {
    auto *sa = dev.getObject(surf_arr_h);
    if (sa && sa->object_type == ANARI_ARRAY1D && sa->memory && sa->array_num_items1 > 0) {
      const auto *surfaces = reinterpret_cast<const uintptr_t *>(sa->memory);
      collect_surfaces_from_array(dev, surfaces, sa->array_num_items1, all_surfaces);
    }
  }

  // ── Collect instanced surfaces from world.instance array ──
  uintptr_t inst_arr_h = 0;
  if (read_handle(wo, "instance", inst_arr_h) && inst_arr_h != 0) {
    auto *ia = dev.getObject(inst_arr_h);
    if (ia && ia->object_type == ANARI_ARRAY1D && ia->memory && ia->array_num_items1 > 0) {
      const auto *instances = reinterpret_cast<const uintptr_t *>(ia->memory);
      for (uint64_t ii = 0; ii < ia->array_num_items1; ++ii) {
        auto *inst_obj = dev.getObject(instances[ii]);
        if (!inst_obj || inst_obj->object_type != ANARI_INSTANCE)
          continue;

        // Read the transform (column-major float[16])
        photon::pt::Mat4 xfm = photon::pt::Mat4::identity();
        float xfm_raw[16];
        if (read_param(inst_obj, "transform", xfm_raw)) {
          xfm = mat4_from_column_major(xfm_raw);
        }

        // Get the group
        uintptr_t group_h = 0;
        if (!read_handle(inst_obj, "group", group_h) || group_h == 0)
          continue;

        auto *group_obj = dev.getObject(group_h);
        if (!group_obj || group_obj->object_type != ANARI_GROUP)
          continue;

        // Get the group's surface array
        uintptr_t group_surf_h = 0;
        if (!read_handle(group_obj, "surface", group_surf_h) || group_surf_h == 0)
          continue;

        auto *gsa = dev.getObject(group_surf_h);
        if (!gsa || gsa->object_type != ANARI_ARRAY1D || !gsa->memory || gsa->array_num_items1 == 0)
          continue;

        const auto *group_surfaces = reinterpret_cast<const uintptr_t *>(gsa->memory);

        // Collect surfaces from this group
        std::vector<SurfaceGeometry> group_sg;
        collect_surfaces_from_array(dev, group_surfaces, gsa->array_num_items1, group_sg);

        // Apply the instance transform to all geometry
        for (auto &sg : group_sg) {
          apply_transform(sg, xfm);
          all_surfaces.push_back(std::move(sg));
        }
      }
    }
  }

  if (all_surfaces.empty())
    return std::nullopt;
  std::vector<photon::pt::Material> matsCpu;
  photon::pt::TriangleMesh mesh;
  merge_surfaces_to_mesh(all_surfaces, mesh, matsCpu);

  if (mesh.triangle_count() == 0)
    return std::nullopt;

  photon::pt::Scene s;
  s.mesh = mesh;
  s.bvh = photon::pt::Bvh::build_cpu(mesh);

  s.material_count = uint32_t(matsCpu.size());
  s.materials = Kokkos::View<photon::pt::Material *>("materials", s.material_count);
  auto mats_h = Kokkos::create_mirror_view(s.materials);
  for (uint32_t i = 0; i < s.material_count; ++i)
    mats_h(i) = matsCpu[i];
  Kokkos::deep_copy(s.materials, mats_h);

  // ── Lights ──
  std::vector<photon::pt::Light> lightsCpu;

  uintptr_t light_arr_h = 0;
  read_handle(wo, "light", light_arr_h);
  const auto *la = (light_arr_h != 0) ? dev.getObject(light_arr_h) : nullptr;
  if (la && la->object_type == ANARI_ARRAY1D && la->memory && la->array_num_items1 > 0) {
    const auto *lightHandles = reinterpret_cast<const uintptr_t *>(la->memory);
    for (uint64_t li = 0; li < la->array_num_items1; ++li) {
      const auto *lo = dev.getObject(lightHandles[li]);
      if (!lo || lo->object_type != ANARI_LIGHT)
        continue;

      const char *lightType = read_string_param(lo, "subtype");
      if (!lightType)
        continue;

      photon::pt::Light light;

      if (std::strcmp(lightType, "directional") == 0) {
        light.type = photon::pt::LightType::Directional;
        photon::pt::Vec3 dir{0.f, -1.f, 0.f};
        read_param(lo, "direction", dir);
        light.direction = dir;

        photon::pt::Vec3 color{1.f, 1.f, 1.f};
        read_param(lo, "color", color);
        light.color = color;

        float intensity = 1.f;
        if (!read_param(lo, "irradiance", intensity))
          read_param(lo, "intensity", intensity);
        light.intensity = intensity;

        lightsCpu.push_back(light);
      } else if (std::strcmp(lightType, "point") == 0) {
        light.type = photon::pt::LightType::Point;
        photon::pt::Vec3 position{0.f, 0.f, 0.f};
        read_param(lo, "position", position);
        light.position = position;

        photon::pt::Vec3 color{1.f, 1.f, 1.f};
        read_param(lo, "color", color);
        light.color = color;

        float intensity = 1.f;
        read_param(lo, "intensity", intensity);
        light.intensity = intensity;

        lightsCpu.push_back(light);
      } else if (std::strcmp(lightType, "quad") == 0) {
        light.type = photon::pt::LightType::Area;
        photon::pt::Vec3 position{0.f, 0.f, 0.f};
        read_param(lo, "position", position);
        light.position = position;

        photon::pt::Vec3 edge1{1.f, 0.f, 0.f};
        read_param(lo, "edge1", edge1);
        light.edge1 = edge1;

        photon::pt::Vec3 edge2{0.f, 0.f, 1.f};
        read_param(lo, "edge2", edge2);
        light.edge2 = edge2;

        photon::pt::Vec3 color{1.f, 1.f, 1.f};
        read_param(lo, "color", color);
        light.color = color;

        float intensity = 1.f;
        read_param(lo, "intensity", intensity);
        light.intensity = intensity;

        photon::pt::Vec3 cr = photon::pt::cross(edge1, edge2);
        light.area = photon::pt::length(cr);
        light.direction = light.area > 0.f ? cr * (1.f / light.area) : photon::pt::Vec3{0.f, -1.f, 0.f};

        lightsCpu.push_back(light);
      }
    }
  }

  if (!lightsCpu.empty()) {
    s.light_count = uint32_t(lightsCpu.size());
    s.lights = Kokkos::View<photon::pt::Light *>("lights", s.light_count);
    auto lights_h = Kokkos::create_mirror_view(s.lights);
    for (uint32_t i = 0; i < s.light_count; ++i)
      lights_h(i) = lightsCpu[i];
    Kokkos::deep_copy(s.lights, lights_h);
  } else {
    s.light_count = 0;
  }

  return s;
}

photon::pt::Camera build_camera_from_anari(ANARIFrame frame, const PhotonDevice &dev)
{
  auto *fo = dev.getObject(uintptr_t(frame));

  uintptr_t cam_h = 0;
  if (fo)
    read_handle(fo, "camera", cam_h);

  photon::pt::Vec3 pos{0.f, 0.f, 5.f};
  photon::pt::Vec3 dir{0.f, 0.f, -1.f};
  photon::pt::Vec3 up{0.f, 1.f, 0.f};

  float fovy_rad = 0.7f;
  float aspect = 1.f;
  float aperture_radius = 0.f;
  float focus_dist = 1.f;

  if (cam_h != 0) {
    auto *co = dev.getObject(cam_h);

    read_param(co, "position", pos);
    read_param(co, "direction", dir);
    read_param(co, "up", up);
    read_param(co, "fovy", fovy_rad);
    read_param(co, "aspect", aspect);
    read_param(co, "apertureRadius", aperture_radius);
    read_param(co, "focusDistance", focus_dist);
  }

  const float fovy_deg = fovy_rad * 180.f / 3.1415926535f;
  return photon::pt::Camera::make_perspective(pos, pos + dir, up, fovy_deg, aspect, 2.f * aperture_radius, focus_dist);
}

}
