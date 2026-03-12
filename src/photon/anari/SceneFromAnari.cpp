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

}

std::optional<photon::pt::Scene> build_scene_from_anari(ANARIWorld world, const PhotonDevice &dev)
{
  if (!world)
    return std::nullopt;

  auto *wo = dev.getObject(uintptr_t(world));
  if (!wo)
    return std::nullopt;

  uintptr_t surf_arr_h = 0;
  if (!read_handle(wo, "surface", surf_arr_h) || surf_arr_h == 0)
    return std::nullopt;

  auto *sa = dev.getObject(surf_arr_h);
  if (!sa || sa->object_type != ANARI_ARRAY1D || !sa->memory || sa->array_num_items1 == 0)
    return std::nullopt;

  const auto *surfaces = reinterpret_cast<const uintptr_t *>(sa->memory);
  const uint64_t nsurf = sa->array_num_items1;

  uint64_t total_tris = 0;
  uint64_t total_verts = 0;
  uint64_t total_indices = 0;

  std::vector<ImportedMaterial> importedMats;
  importedMats.reserve(size_t(nsurf));

  for (uint64_t si = 0; si < nsurf; ++si) {
    auto *so = dev.getObject(surfaces[si]);
    if (!so || so->object_type != ANARI_SURFACE)
      continue;

    uintptr_t geom_h = 0;
    if (!read_handle(so, "geometry", geom_h) || geom_h == 0)
      continue;
    auto *go = dev.getObject(geom_h);
    if (!go || go->object_type != ANARI_GEOMETRY)
      continue;

    const auto *posA = read_array1d(dev, go, "vertex.position", ANARI_FLOAT32_VEC3);
    if (!posA)
      continue;

    uint64_t trisForSurface = 0;

    uintptr_t idx_h = 0;
    read_handle(go, "primitive.index", idx_h);
    const auto *idxA = idx_h ? dev.getObject(idx_h) : nullptr;

    if (idxA && idxA->object_type == ANARI_ARRAY1D && idxA->memory) {
      if (idxA->array_element_type == ANARI_UINT32_VEC3) {
        trisForSurface = idxA->array_num_items1;
      } else if (idxA->array_element_type == ANARI_UINT32_VEC4) {
        trisForSurface = 2 * idxA->array_num_items1;
      } else {
        continue;
      }
    } else {
      const char *subtype = read_string_param(go, "subtype");
      const uint64_t nv = posA->array_num_items1;
      if (subtype && std::strcmp(subtype, "quad") == 0)
        trisForSurface = 2 * (nv / 4);
      else
        trisForSurface = (nv / 3);
    }

    if (trisForSurface == 0)
      continue;

    importedMats.push_back(import_material(dev, so));

    total_tris += trisForSurface;
    total_verts += 3 * trisForSurface;
    total_indices += 3 * trisForSurface;
  }

  if (total_tris == 0)
    return std::nullopt;

  photon::pt::TriangleMesh mesh;
  mesh.positions = Kokkos::View<photon::pt::Vec3 *>("pos", total_verts);
  mesh.indices = Kokkos::View<photon::pt::u32 *>("idx", total_indices);
  mesh.material_ids = Kokkos::View<photon::pt::u32 *>("mat_id", total_tris);
  mesh.albedo_per_prim = Kokkos::View<photon::pt::Vec3 *>("alb", total_tris);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto mat_id_h = Kokkos::create_mirror_view(mesh.material_ids);
  auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);

  uint64_t vert_off = 0;
  uint64_t idx_off = 0;
  uint64_t tri_off = 0;

  std::vector<photon::pt::Material> matsCpu;
  matsCpu.reserve(importedMats.size());

  size_t matCursor = 0;

  for (uint64_t si = 0; si < nsurf; ++si) {
    auto *so = dev.getObject(surfaces[si]);
    if (!so || so->object_type != ANARI_SURFACE)
      continue;

    uintptr_t geom_h = 0;
    if (!read_handle(so, "geometry", geom_h) || geom_h == 0)
      continue;
    auto *go = dev.getObject(geom_h);
    if (!go || go->object_type != ANARI_GEOMETRY)
      continue;

    const auto *posA = read_array1d(dev, go, "vertex.position", ANARI_FLOAT32_VEC3);
    if (!posA)
      continue;

    const float *pos = reinterpret_cast<const float *>(posA->memory);

    const auto *colA = read_array1d(dev, go, "vertex.color", ANARI_FLOAT32_VEC4);
    const float *col = colA ? reinterpret_cast<const float *>(colA->memory) : nullptr;

    const auto *nrmA = read_array1d(dev, go, "vertex.normal", ANARI_FLOAT32_VEC3);
    const float *nrm = nrmA ? reinterpret_cast<const float *>(nrmA->memory) : nullptr;

    uintptr_t idxHandle = 0;
    read_handle(go, "primitive.index", idxHandle);
    const auto *idxA = idxHandle ? dev.getObject(idxHandle) : nullptr;

    std::vector<TriIdx> tris;

    if (idxA && idxA->object_type == ANARI_ARRAY1D && idxA->memory) {
      if (idxA->array_element_type == ANARI_UINT32_VEC3) {
        tris.resize(idxA->array_num_items1);
        const uint32_t *src = reinterpret_cast<const uint32_t *>(idxA->memory);
        for (uint64_t i = 0; i < idxA->array_num_items1; ++i) {
          tris[i] = {src[3 * i + 0], src[3 * i + 1], src[3 * i + 2]};
        }
      } else if (idxA->array_element_type == ANARI_UINT32_VEC4) {
        tris.resize(2 * idxA->array_num_items1);
        const uint32_t *src = reinterpret_cast<const uint32_t *>(idxA->memory);
        for (uint64_t i = 0; i < idxA->array_num_items1; ++i) {
          const uint32_t a = src[4 * i + 0];
          const uint32_t b = src[4 * i + 1];
          const uint32_t c = src[4 * i + 2];
          const uint32_t d = src[4 * i + 3];
          tris[2 * i + 0] = {a, b, c};
          tris[2 * i + 1] = {a, c, d};
        }
      } else {
        continue;
      }
    } else {
      const char *subtype = read_string_param(go, "subtype");
      const uint64_t nv = posA->array_num_items1;
      if (subtype && std::strcmp(subtype, "quad") == 0) {
        const uint64_t nq = nv / 4;
        tris.resize(2 * nq);
        for (uint64_t q = 0; q < nq; ++q) {
          const uint32_t base = uint32_t(4 * q);
          tris[2 * q + 0] = {base + 0, base + 1, base + 2};
          tris[2 * q + 1] = {base + 0, base + 2, base + 3};
        }
      } else {
        const uint64_t nt = nv / 3;
        tris.resize(nt);
        for (uint64_t t = 0; t < nt; ++t) {
          const uint32_t base = uint32_t(3 * t);
          tris[t] = {base + 0, base + 1, base + 2};
        }
      }
    }

    if (tris.empty())
      continue;

    const ImportedMaterial &im = importedMats[matCursor++];

    if (im.use_vertex_color && col) {
      for (const auto &t : tris) {
        const uint32_t vi0 = t.a;
        const uint32_t vi1 = t.b;
        const uint32_t vi2 = t.c;

        const float *p0 = pos + 3 * size_t(vi0);
        const float *p1 = pos + 3 * size_t(vi1);
        const float *p2 = pos + 3 * size_t(vi2);

        pos_h(vert_off + 0) = v3_from_floats(p0);
        pos_h(vert_off + 1) = v3_from_floats(p1);
        pos_h(vert_off + 2) = v3_from_floats(p2);

        idx_h(idx_off + 0) = photon::pt::u32(vert_off + 0);
        idx_h(idx_off + 1) = photon::pt::u32(vert_off + 1);
        idx_h(idx_off + 2) = photon::pt::u32(vert_off + 2);

        const float *c0 = col + 4 * size_t(vi0);
        const float *c1 = col + 4 * size_t(vi1);
        const float *c2 = col + 4 * size_t(vi2);
        const photon::pt::Vec3 a0 = v3_from_vec4(c0);
        const photon::pt::Vec3 a1 = v3_from_vec4(c1);
        const photon::pt::Vec3 a2 = v3_from_vec4(c2);
        const photon::pt::Vec3 alb = (a0 + a1 + a2) * (1.f / 3.f);

        const uint32_t matId = uint32_t(matsCpu.size());
        photon::pt::Material triMat = im.mat;
        triMat.base_color = alb;
        matsCpu.push_back(triMat);

        mat_id_h(tri_off) = matId;
        alb_h(tri_off) = alb;

        vert_off += 3;
        idx_off += 3;
        tri_off += 1;
      }
    } else {
      const uint32_t matId = uint32_t(matsCpu.size());
      matsCpu.push_back(im.mat);

      for (const auto &t : tris) {
        const uint32_t vi0 = t.a;
        const uint32_t vi1 = t.b;
        const uint32_t vi2 = t.c;

        const float *p0 = pos + 3 * size_t(vi0);
        const float *p1 = pos + 3 * size_t(vi1);
        const float *p2 = pos + 3 * size_t(vi2);

        pos_h(vert_off + 0) = v3_from_floats(p0);
        pos_h(vert_off + 1) = v3_from_floats(p1);
        pos_h(vert_off + 2) = v3_from_floats(p2);

        idx_h(idx_off + 0) = photon::pt::u32(vert_off + 0);
        idx_h(idx_off + 1) = photon::pt::u32(vert_off + 1);
        idx_h(idx_off + 2) = photon::pt::u32(vert_off + 2);

        mat_id_h(tri_off) = matId;
        alb_h(tri_off) = im.mat.base_color;

        vert_off += 3;
        idx_off += 3;
        tri_off += 1;
      }
    }

    (void)nrm;
  }

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.material_ids, mat_id_h);
  Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);

  photon::pt::Scene s;
  s.mesh = mesh;
  s.bvh = photon::pt::Bvh::build_cpu(mesh);

  s.material_count = uint32_t(matsCpu.size());
  s.materials = Kokkos::View<photon::pt::Material *>("materials", s.material_count);
  auto mats_h = Kokkos::create_mirror_view(s.materials);
  for (uint32_t i = 0; i < s.material_count; ++i)
    mats_h(i) = matsCpu[i];
  Kokkos::deep_copy(s.materials, mats_h);

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
