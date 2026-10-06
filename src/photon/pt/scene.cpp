#include "photon/pt/scene.h"

#include <cstdio>

namespace photon::pt {

TriangleMesh object_mesh_to_triangle_mesh(const ObjectMesh &obj)
{  const uint64_t nv = obj.positions.size() / 3;
  const uint64_t nt = obj.indices.size() / 3;
  const bool has_normals = obj.normals.size() >= obj.positions.size();
  const bool has_uvs = obj.uvs.size() >= nv * 2;

  TriangleMesh tm;
  tm.positions    = Kokkos::View<Vec3 *>("obj_pos", nv);
  tm.indices      = Kokkos::View<u32 *>("obj_idx", nt * 3);
  tm.material_ids = Kokkos::View<u32 *>("obj_mat", nt);
  if (has_normals)
    tm.normals    = Kokkos::View<Vec3 *>("obj_nrm", nv);
  if (has_uvs)
    tm.texcoords  = Kokkos::View<Vec2 *>("obj_uv", nv);

  auto ph = Kokkos::create_mirror_view(tm.positions);
  auto ih = Kokkos::create_mirror_view(tm.indices);
  auto mh = Kokkos::create_mirror_view(tm.material_ids);
  decltype(Kokkos::create_mirror_view(tm.normals)) nh;
  decltype(Kokkos::create_mirror_view(tm.texcoords)) uvh;

  for (uint64_t v = 0; v < nv; ++v)
    ph(v) = {obj.positions[v * 3], obj.positions[v * 3 + 1], obj.positions[v * 3 + 2]};
  for (uint64_t i = 0; i < nt * 3; ++i)
    ih(i) = u32(obj.indices[i]);
  for (uint64_t t = 0; t < nt; ++t)
    mh(t) = obj.material_id;
  if (has_normals) {
    nh = Kokkos::create_mirror_view(tm.normals);
    for (uint64_t v = 0; v < nv; ++v)
      nh(v) = {obj.normals[v * 3], obj.normals[v * 3 + 1], obj.normals[v * 3 + 2]};
  }
  if (has_uvs) {
    uvh = Kokkos::create_mirror_view(tm.texcoords);
    for (uint64_t v = 0; v < nv; ++v)
      uvh(v) = {obj.uvs[v * 2], obj.uvs[v * 2 + 1]};
  }

  Kokkos::deep_copy(tm.positions, ph);
  Kokkos::deep_copy(tm.indices, ih);
  Kokkos::deep_copy(tm.material_ids, mh);
  if (has_normals) Kokkos::deep_copy(tm.normals, nh);
  if (has_uvs) Kokkos::deep_copy(tm.texcoords, uvh);

  return tm;
}

namespace {

// Accumulator for the flattening pass — plain host vectors, uploaded once.
struct FlatMesh {
  std::vector<Vec3> positions;
  std::vector<u32> indices;
  std::vector<Vec3> normals;
  std::vector<Vec2> texcoords;
  std::vector<u32> material_ids;

  void add_triangle(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2,
                    const Vec3 &n0, const Vec3 &n1, const Vec3 &n2,
                    const Vec2 uv0, const Vec2 uv1, const Vec2 uv2,
                    bool has_uvs, u32 mat_id)
  {
    const u32 base = u32(positions.size());
    positions.push_back(p0);
    positions.push_back(p1);
    positions.push_back(p2);
    normals.push_back(n0);
    normals.push_back(n1);
    normals.push_back(n2);
    if (has_uvs) {
      texcoords.push_back(uv0);
      texcoords.push_back(uv1);
      texcoords.push_back(uv2);
    }
    indices.push_back(base + 0);
    indices.push_back(base + 1);
    indices.push_back(base + 2);
    material_ids.push_back(mat_id);
  }
};

} // anonymous namespace

FlatInstancedMesh flatten_instanced_geometry(const Scene &scene,
                                             const InstancedGeometry &instanced)
{
  FlatMesh flat;
  FlatInstancedMesh out;

  // Scene-level emissive tags carry over: their triangle indices stay valid
  // because the scene mesh is copied first, in order.
  if (scene.emissive_count > 0) {
    auto ids_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.emissive_prim_ids);
    auto areas_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.emissive_prim_areas);
    for (u32 i = 0; i < scene.emissive_count; ++i) {
      out.emissive_prim_ids.push_back(ids_h(i));
      out.emissive_prim_areas.push_back(areas_h(i));
    }
  }

  // Host copy of materials for emissive detection on instance triangles.
  const bool have_materials = scene.materials.extent(0) > 0;
  auto mats_h = have_materials
      ? Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.materials)
      : decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.materials)){};
  auto is_emissive = [&](u32 mat_id) {
    return have_materials && mat_id < mats_h.extent(0) &&
        material_is_emissive(mats_h(mat_id));
  };

  // Scene-level mesh first (if any), copied verbatim.
  {
    auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.positions);
    auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.indices);
    const uint64_t nt = scene.mesh.triangle_count();
    const bool has_normals = scene.mesh.has_normals();
    const bool has_uvs = scene.mesh.has_texcoords();
    const bool has_mats = scene.mesh.has_material_ids();
    auto nrm_h = has_normals
        ? Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.normals)
        : decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.normals)){};
    auto uv_h = has_uvs
        ? Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.texcoords)
        : decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.texcoords)){};
    auto mat_h = has_mats
        ? Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.material_ids)
        : decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.mesh.material_ids)){};

    for (uint64_t t = 0; t < nt; ++t) {
      const u32 i0 = idx_h(t * 3 + 0), i1 = idx_h(t * 3 + 1), i2 = idx_h(t * 3 + 2);
      Vec3 n0 = has_normals ? nrm_h(i0) : normalize(cross(pos_h(i1) - pos_h(i0), pos_h(i2) - pos_h(i0)));
      flat.add_triangle(pos_h(i0), pos_h(i1), pos_h(i2), n0, n0, n0,
          has_uvs ? uv_h(i0) : Vec2{0.f, 0.f},
          has_uvs ? uv_h(i1) : Vec2{0.f, 0.f},
          has_uvs ? uv_h(i2) : Vec2{0.f, 0.f},
          has_uvs, has_mats ? mat_h(t) : 0u);
    }
  }

  // Then each instance, transformed into world space.
  for (const auto &inst : instanced.instances) {
    if (inst.object_id >= instanced.objects.size()) continue;
    const ObjectMesh &obj = instanced.objects[inst.object_id];
    if (obj.indices.empty()) continue;

    const Mat4 xfm = mat4_from_column_major(inst.transform);
    const uint64_t nv = obj.positions.size() / 3;
    const uint64_t nt = obj.indices.size() / 3;
    const bool has_normals = obj.normals.size() >= obj.positions.size();
    const bool has_uvs = obj.uvs.size() >= nv * 2;

    for (uint64_t t = 0; t < nt; ++t) {
      const int i0 = obj.indices[t * 3 + 0];
      const int i1 = obj.indices[t * 3 + 1];
      const int i2 = obj.indices[t * 3 + 2];

      Vec3 p0 = xfm.transform_point({obj.positions[i0 * 3], obj.positions[i0 * 3 + 1], obj.positions[i0 * 3 + 2]});
      Vec3 p1 = xfm.transform_point({obj.positions[i1 * 3], obj.positions[i1 * 3 + 1], obj.positions[i1 * 3 + 2]});
      Vec3 p2 = xfm.transform_point({obj.positions[i2 * 3], obj.positions[i2 * 3 + 1], obj.positions[i2 * 3 + 2]});

      Vec3 n0, n1, n2;
      if (has_normals) {
        n0 = normalize(xfm.transform_direction({obj.normals[i0 * 3], obj.normals[i0 * 3 + 1], obj.normals[i0 * 3 + 2]}));
        n1 = normalize(xfm.transform_direction({obj.normals[i1 * 3], obj.normals[i1 * 3 + 1], obj.normals[i1 * 3 + 2]}));
        n2 = normalize(xfm.transform_direction({obj.normals[i2 * 3], obj.normals[i2 * 3 + 1], obj.normals[i2 * 3 + 2]}));
      } else {
        Vec3 face_n = normalize(cross(p1 - p0, p2 - p0));
        n0 = n1 = n2 = face_n;
      }

      if (is_emissive(obj.material_id)) {
        out.emissive_prim_ids.push_back(u32(flat.positions.size() / 3));
        out.emissive_prim_areas.push_back(0.5f * length(cross(p1 - p0, p2 - p0)));
      }

      flat.add_triangle(p0, p1, p2, n0, n1, n2,
          has_uvs ? Vec2{obj.uvs[i0 * 2], obj.uvs[i0 * 2 + 1]} : Vec2{0.f, 0.f},
          has_uvs ? Vec2{obj.uvs[i1 * 2], obj.uvs[i1 * 2 + 1]} : Vec2{0.f, 0.f},
          has_uvs ? Vec2{obj.uvs[i2 * 2], obj.uvs[i2 * 2 + 1]} : Vec2{0.f, 0.f},
          has_uvs, obj.material_id);
    }
  }

  const uint64_t nv = flat.positions.size();
  const uint64_t nt = flat.indices.size() / 3;

  TriangleMesh tm;
  tm.positions    = Kokkos::View<Vec3 *>("flat_pos", nv);
  tm.indices      = Kokkos::View<u32 *>("flat_idx", nt * 3);
  tm.normals      = Kokkos::View<Vec3 *>("flat_nrm", nv);
  tm.material_ids = Kokkos::View<u32 *>("flat_mat", nt);
  if (flat.texcoords.size() >= nv * 2)
    tm.texcoords  = Kokkos::View<Vec2 *>("flat_uv", nv);

  auto pos_h = Kokkos::create_mirror_view(tm.positions);
  auto idx_h = Kokkos::create_mirror_view(tm.indices);
  auto nrm_h = Kokkos::create_mirror_view(tm.normals);
  auto mat_h = Kokkos::create_mirror_view(tm.material_ids);
  for (uint64_t v = 0; v < nv; ++v) {
    pos_h(v) = flat.positions[v];
    nrm_h(v) = flat.normals[v];
  }
  for (uint64_t i = 0; i < nt * 3; ++i)
    idx_h(i) = flat.indices[i];
  for (uint64_t t = 0; t < nt; ++t)
    mat_h(t) = flat.material_ids[t];

  Kokkos::deep_copy(tm.positions, pos_h);
  Kokkos::deep_copy(tm.indices, idx_h);
  Kokkos::deep_copy(tm.normals, nrm_h);
  Kokkos::deep_copy(tm.material_ids, mat_h);
  if (tm.texcoords.data()) {
    auto uv_h = Kokkos::create_mirror_view(tm.texcoords);
    for (uint64_t v = 0; v < nv; ++v)
      uv_h(v) = flat.texcoords[v];
    Kokkos::deep_copy(tm.texcoords, uv_h);
  }

  out.mesh = tm;
  return out;
}

} // namespace photon::pt
