#include "photon/anari/SceneFromAnari.h"

#include <cstring>

#include <Kokkos_Core.hpp>

#include "photon/anari/PhotonDevice.h"

#include <optional>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/camera.h"
#include "photon/pt/math.h"
#include "photon/pt/scene.h"

namespace photon::anari_device {

namespace {

template<typename T>
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

}

std::optional<photon::pt::Scene> build_scene_from_anari(ANARIWorld world, const PhotonDevice &dev)
{
  if (!world)
    return std::nullopt;

  auto *wo = dev.getObject(uintptr_t(world));
  if (!wo)
    return std::nullopt;

  uintptr_t surfaces_arr = 0;
  if (!read_handle(wo, "surface", surfaces_arr) || surfaces_arr == 0)
    return std::nullopt;

  auto *sa = dev.getObject(surfaces_arr);
  if (!sa || sa->object_type != ANARI_ARRAY1D || !sa->memory || sa->array_num_items1 == 0)
    return std::nullopt;

  const auto *surfaces = reinterpret_cast<const uintptr_t *>(sa->memory);
  const uint64_t nsurf = sa->array_num_items1;

  uint64_t total_pos = 0;
  uint64_t total_idx = 0;
  uint64_t total_prims = 0;

  for (uint64_t si = 0; si < nsurf; ++si) {
    auto *so = dev.getObject(surfaces[si]);
    if (!so)
      continue;

    uintptr_t geom_h = 0;
    if (!read_handle(so, "geometry", geom_h) || geom_h == 0)
      continue;

    auto *go = dev.getObject(geom_h);
    if (!go)
      continue;

    const char *subtype = read_string_param(go, "subtype");

    uintptr_t vtx_h = 0;
    if (!read_handle(go, "vertex.position", vtx_h) || vtx_h == 0)
      continue;

    auto *va = dev.getObject(vtx_h);
    if (!va || !va->memory)
      continue;

    const uint64_t nv = va->array_num_items1;

    uintptr_t idx_hnd = 0;
    read_handle(go, "primitive.index", idx_hnd);
    auto *ia = idx_hnd ? dev.getObject(idx_hnd) : nullptr;

    uint64_t nprim = 0;
    if (ia && ia->memory)
      nprim = ia->array_num_items1;
    else
      nprim = (subtype && std::strcmp(subtype, "quad") == 0) ? (nv / 4) : (nv / 3);

    if (subtype && std::strcmp(subtype, "quad") == 0) {
      total_pos += 4 * nprim;
      total_idx += 6 * nprim;
      total_prims += 2 * nprim;
    } else {
      total_pos += 3 * nprim;
      total_idx += 3 * nprim;
      total_prims += 1 * nprim;
    }
  }

  if (total_pos == 0 || total_idx == 0)
    return std::nullopt;

  photon::pt::TriangleMesh mesh;
  mesh.positions = Kokkos::View<photon::pt::Vec3 *>("pos", total_pos);
  mesh.indices = Kokkos::View<photon::pt::u32 *>("idx", total_idx);
  mesh.material_ids = Kokkos::View<photon::pt::u32 *>("mat_id", total_prims);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto mat_id_h = Kokkos::create_mirror_view(mesh.material_ids);

  uint64_t pos_off = 0;
  uint64_t idx_off = 0;
  uint64_t prim_off = 0;

  for (uint64_t si = 0; si < nsurf; ++si) {
    auto *so = dev.getObject(surfaces[si]);
    if (!so)
      continue;

    uintptr_t geom_h = 0;
    if (!read_handle(so, "geometry", geom_h) || geom_h == 0)
      continue;

    auto *go = dev.getObject(geom_h);
    if (!go)
      continue;

    const char *subtype = read_string_param(go, "subtype");

    uintptr_t vtx_h = 0;
    if (!read_handle(go, "vertex.position", vtx_h) || vtx_h == 0)
      continue;

    auto *va = dev.getObject(vtx_h);
    if (!va || !va->memory)
      continue;

    const float *v = reinterpret_cast<const float *>(va->memory);

    uintptr_t idx_hnd = 0;
    read_handle(go, "primitive.index", idx_hnd);
    auto *ia = idx_hnd ? dev.getObject(idx_hnd) : nullptr;

    const uint64_t nprim = (ia && ia->memory) ? ia->array_num_items1
                                              : ((subtype && std::strcmp(subtype, "quad") == 0) ? (va->array_num_items1 / 4)
                                                                                                 : (va->array_num_items1 / 3));

    for (uint64_t pi = 0; pi < nprim; ++pi) {
      if (subtype && std::strcmp(subtype, "quad") == 0) {
        const float *vv = v + 12 * pi;
        pos_h(pos_off + 0) = {vv[0], vv[1], vv[2]};
        pos_h(pos_off + 1) = {vv[3], vv[4], vv[5]};
        pos_h(pos_off + 2) = {vv[6], vv[7], vv[8]};
        pos_h(pos_off + 3) = {vv[9], vv[10], vv[11]};

        uint32_t q[4] = {0, 1, 2, 3};
        if (ia && ia->memory) {
          const uint32_t *qi = reinterpret_cast<const uint32_t *>(ia->memory) + 4 * pi;
          q[0] = qi[0];
          q[1] = qi[1];
          q[2] = qi[2];
          q[3] = qi[3];
        }

        idx_h(idx_off + 0) = photon::pt::u32(pos_off + q[0]);
        idx_h(idx_off + 1) = photon::pt::u32(pos_off + q[1]);
        idx_h(idx_off + 2) = photon::pt::u32(pos_off + q[2]);
        idx_h(idx_off + 3) = photon::pt::u32(pos_off + q[0]);
        idx_h(idx_off + 4) = photon::pt::u32(pos_off + q[2]);
        idx_h(idx_off + 5) = photon::pt::u32(pos_off + q[3]);

        mat_id_h(prim_off + 0) = 0;
        mat_id_h(prim_off + 1) = 0;

        pos_off += 4;
        idx_off += 6;
        prim_off += 2;
      } else {
        const float *vv = v + 9 * pi;
        pos_h(pos_off + 0) = {vv[0], vv[1], vv[2]};
        pos_h(pos_off + 1) = {vv[3], vv[4], vv[5]};
        pos_h(pos_off + 2) = {vv[6], vv[7], vv[8]};

        uint32_t t[3] = {0, 1, 2};
        if (ia && ia->memory) {
          const uint32_t *ti = reinterpret_cast<const uint32_t *>(ia->memory) + 3 * pi;
          t[0] = ti[0];
          t[1] = ti[1];
          t[2] = ti[2];
        }

        idx_h(idx_off + 0) = photon::pt::u32(pos_off + t[0]);
        idx_h(idx_off + 1) = photon::pt::u32(pos_off + t[1]);
        idx_h(idx_off + 2) = photon::pt::u32(pos_off + t[2]);

        mat_id_h(prim_off) = 0;

        pos_off += 3;
        idx_off += 3;
        prim_off += 1;
      }
    }
  }

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.material_ids, mat_id_h);

  photon::pt::Scene s;
  s.mesh = mesh;
  s.bvh = photon::pt::Bvh::build_cpu(mesh);

  auto mats_h = Kokkos::View<photon::pt::Material *>("materials", 1);
  auto mats_host = Kokkos::create_mirror_view(mats_h);
  mats_host(0) = photon::pt::Material{};
  Kokkos::deep_copy(mats_h, mats_host);
  s.materials = mats_h;
  s.material_count = 1;

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
