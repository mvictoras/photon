#include "photon/pbrt/pbrt_to_photon.h"
#include "photon/pbrt/image_loader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/math.h"
#include "photon/pt/math_mat4.h"
#include "photon/pt/scene.h"

namespace photon::pbrt {

using namespace photon::pt;

static Mat4 mat4_from_pbrt_column_major(const float *cm)
{
  Mat4 m{};
  for (int col = 0; col < 4; ++col)
    for (int row = 0; row < 4; ++row)
      m.m[row][col] = cm[col * 4 + row];
  return m;
}

static Vec3 transform_point(const Mat4 &m, const Vec3 &p)
{
  return m.transform_point(p);
}

static Vec3 transform_direction(const Mat4 &m, const Vec3 &d)
{
  return m.transform_direction(d);
}

static Vec3 sample_texture(const PbrtTexture &tex, float u, float v)
{
  u = u - std::floor(u);
  v = v - std::floor(v);
  int px = int(u * float(tex.width)) % tex.width;
  int py = int((1.f - v) * float(tex.height)) % tex.height;
  if (px < 0) px += tex.width;
  if (py < 0) py += tex.height;
  const size_t idx = (size_t(py) * size_t(tex.width) + size_t(px)) * 3;
  if (idx + 2 >= tex.data.size())
    return {0.5f, 0.5f, 0.5f};
  return {tex.data[idx], tex.data[idx + 1], tex.data[idx + 2]};
}

ConvertedScene convert_pbrt_scene(const PbrtScene &pbrt, const std::string &base_dir)
{
  ConvertedScene result;

  auto &textures = const_cast<std::map<std::string, PbrtTexture> &>(pbrt.textures);
  for (auto &[name, tex] : textures) {
    if (tex.class_type == "imagemap" && !tex.filename.empty() && tex.data.empty()) {
      std::string path = base_dir + tex.filename;
      load_image(path, tex);
    }
  }

  uint64_t total_tris = 0;
  for (const auto &mesh : pbrt.meshes)
    total_tris += mesh.indices.size() / 3;

  if (total_tris == 0) {
    std::fprintf(stderr, "pbrt scene has no triangles\n");
    return result;
  }

  const uint64_t total_verts = total_tris * 3;

  bool any_textured = false;
  for (const auto &mesh : pbrt.meshes) {
    auto mit = pbrt.named_materials.find(mesh.material_name);
    if (mit != pbrt.named_materials.end() && !mit->second.reflectance_texture.empty()) {
      auto tit = pbrt.textures.find(mit->second.reflectance_texture);
      if (tit != pbrt.textures.end() && !tit->second.data.empty()) {
        any_textured = true;
        break;
      }
    }
  }

  TriangleMesh tm;
  tm.positions = Kokkos::View<Vec3 *>("pos", total_verts);
  tm.indices = Kokkos::View<u32 *>("idx", total_verts);
  tm.material_ids = Kokkos::View<u32 *>("mat_id", total_tris);
  tm.albedo_per_prim = Kokkos::View<Vec3 *>("alb", total_tris);
  tm.normals = Kokkos::View<Vec3 *>("normals", total_verts);
  if (any_textured)
    tm.vertex_colors = Kokkos::View<Vec3 *>("vcol", total_verts);

  auto pos_h = Kokkos::create_mirror_view(tm.positions);
  auto idx_h = Kokkos::create_mirror_view(tm.indices);
  auto mat_id_h = Kokkos::create_mirror_view(tm.material_ids);
  auto alb_h = Kokkos::create_mirror_view(tm.albedo_per_prim);
  auto nrm_h = Kokkos::create_mirror_view(tm.normals);
  decltype(Kokkos::create_mirror_view(tm.vertex_colors)) vcol_h;
  if (any_textured)
    vcol_h = Kokkos::create_mirror_view(tm.vertex_colors);

  std::vector<Material> materials_cpu;
  std::vector<Light> lights_cpu;
  std::vector<u32> emissive_prim_ids;
  std::vector<f32> emissive_prim_areas;

  std::map<std::string, u32> mat_name_to_id;

  for (const auto &[name, pmat] : pbrt.named_materials) {
    u32 id = u32(materials_cpu.size());
    mat_name_to_id[name] = id;

    Material m{};
    if (pmat.type == "diffuse") {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.roughness = 1.f;
      m.metallic = 0.f;
    } else if (pmat.type == "conductor") {
      m.base_color = {pmat.eta.x, pmat.eta.y, pmat.eta.z};
      m.metallic = 1.f;
      float r = pmat.roughness;
      if (pmat.uroughness >= 0.f)
        r = (pmat.uroughness + (pmat.vroughness >= 0.f ? pmat.vroughness : pmat.uroughness)) * 0.5f;
      m.roughness = r;
    } else if (pmat.type == "dielectric") {
      m.base_color = {1.f, 1.f, 1.f};
      m.ior = pmat.eta_scalar;
      m.transmission = 1.f;
      float r = pmat.roughness;
      if (pmat.uroughness >= 0.f)
        r = (pmat.uroughness + (pmat.vroughness >= 0.f ? pmat.vroughness : pmat.uroughness)) * 0.5f;
      m.roughness = r;
    } else if (pmat.type == "coateddiffuse") {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.metallic = 0.f;
      float r = pmat.roughness;
      if (pmat.uroughness >= 0.f)
        r = (pmat.uroughness + (pmat.vroughness >= 0.f ? pmat.vroughness : pmat.uroughness)) * 0.5f;
      m.roughness = r;
      m.clearcoat = 1.f;
      m.clearcoat_roughness = r * 0.5f;
    } else {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.roughness = pmat.roughness;
    }
    materials_cpu.push_back(m);
  }

  if (materials_cpu.empty()) {
    Material fallback{};
    fallback.base_color = {0.5f, 0.5f, 0.5f};
    fallback.roughness = 1.f;
    materials_cpu.push_back(fallback);
    mat_name_to_id["__default__"] = 0;
  }

  uint64_t vert_off = 0;
  uint64_t tri_off = 0;

  for (const auto &mesh : pbrt.meshes) {
    Mat4 xfm = mat4_from_pbrt_column_major(mesh.transform);

    u32 mat_id = 0;
    auto it = mat_name_to_id.find(mesh.material_name);
    if (it != mat_name_to_id.end())
      mat_id = it->second;

    const PbrtTexture *tex_ptr = nullptr;
    {
      auto mit = pbrt.named_materials.find(mesh.material_name);
      if (mit != pbrt.named_materials.end() && !mit->second.reflectance_texture.empty()) {
        auto tit = pbrt.textures.find(mit->second.reflectance_texture);
        if (tit != pbrt.textures.end() && !tit->second.data.empty())
          tex_ptr = &tit->second;
      }
    }
    const bool has_uvs = mesh.uvs.size() >= (mesh.positions.size() / 3) * 2;

    if (mesh.is_emissive) {
      Material emit_mat{};
      emit_mat.base_color = {0.f, 0.f, 0.f};
      emit_mat.emission = {mesh.emission.x, mesh.emission.y, mesh.emission.z};
      emit_mat.emission_strength = 1.f;
      emit_mat.roughness = 1.f;
      mat_id = u32(materials_cpu.size());
      materials_cpu.push_back(emit_mat);
    }

    const size_t ntris = mesh.indices.size() / 3;
    const bool has_normals = mesh.normals.size() >= mesh.positions.size();

    for (size_t t = 0; t < ntris; ++t) {
      const int i0 = mesh.indices[t * 3 + 0];
      const int i1 = mesh.indices[t * 3 + 1];
      const int i2 = mesh.indices[t * 3 + 2];

      Vec3 p0 = {mesh.positions[i0*3], mesh.positions[i0*3+1], mesh.positions[i0*3+2]};
      Vec3 p1 = {mesh.positions[i1*3], mesh.positions[i1*3+1], mesh.positions[i1*3+2]};
      Vec3 p2 = {mesh.positions[i2*3], mesh.positions[i2*3+1], mesh.positions[i2*3+2]};

      p0 = transform_point(xfm, p0);
      p1 = transform_point(xfm, p1);
      p2 = transform_point(xfm, p2);

      pos_h(vert_off + 0) = p0;
      pos_h(vert_off + 1) = p1;
      pos_h(vert_off + 2) = p2;

      idx_h(vert_off + 0) = u32(vert_off + 0);
      idx_h(vert_off + 1) = u32(vert_off + 1);
      idx_h(vert_off + 2) = u32(vert_off + 2);

      if (has_normals) {
        Vec3 n0 = {mesh.normals[i0*3], mesh.normals[i0*3+1], mesh.normals[i0*3+2]};
        Vec3 n1 = {mesh.normals[i1*3], mesh.normals[i1*3+1], mesh.normals[i1*3+2]};
        Vec3 n2 = {mesh.normals[i2*3], mesh.normals[i2*3+1], mesh.normals[i2*3+2]};
        nrm_h(vert_off + 0) = normalize(transform_direction(xfm, n0));
        nrm_h(vert_off + 1) = normalize(transform_direction(xfm, n1));
        nrm_h(vert_off + 2) = normalize(transform_direction(xfm, n2));
      } else {
        Vec3 face_n = normalize(cross(p1 - p0, p2 - p0));
        nrm_h(vert_off + 0) = face_n;
        nrm_h(vert_off + 1) = face_n;
        nrm_h(vert_off + 2) = face_n;
      }

      if (tex_ptr && has_uvs) {
        float u0 = mesh.uvs[i0 * 2], v0 = mesh.uvs[i0 * 2 + 1];
        float u1 = mesh.uvs[i1 * 2], v1 = mesh.uvs[i1 * 2 + 1];
        float u2 = mesh.uvs[i2 * 2], v2 = mesh.uvs[i2 * 2 + 1];
        vcol_h(vert_off + 0) = sample_texture(*tex_ptr, u0, v0);
        vcol_h(vert_off + 1) = sample_texture(*tex_ptr, u1, v1);
        vcol_h(vert_off + 2) = sample_texture(*tex_ptr, u2, v2);
      }

      mat_id_h(tri_off) = mat_id;
      alb_h(tri_off) = materials_cpu[mat_id].base_color;

      if (mesh.is_emissive) {
        emissive_prim_ids.push_back(u32(tri_off));
        f32 area = 0.5f * length(cross(p1 - p0, p2 - p0));
        emissive_prim_areas.push_back(area);
      }

      vert_off += 3;
      tri_off += 1;
    }
  }

  Kokkos::deep_copy(tm.positions, pos_h);
  Kokkos::deep_copy(tm.indices, idx_h);
  Kokkos::deep_copy(tm.material_ids, mat_id_h);
  Kokkos::deep_copy(tm.albedo_per_prim, alb_h);
  Kokkos::deep_copy(tm.normals, nrm_h);
  if (any_textured)
    Kokkos::deep_copy(tm.vertex_colors, vcol_h);

  Scene &scene = result.scene;
  scene.mesh = tm;
  scene.bvh = Bvh::build_cpu(tm);

  scene.material_count = u32(materials_cpu.size());
  {
    Kokkos::View<Material *, Kokkos::HostSpace> mats_host("mats_host", scene.material_count);
    for (u32 i = 0; i < scene.material_count; ++i)
      mats_host(i) = materials_cpu[i];
    scene.materials = Kokkos::View<Material *>("materials", scene.material_count);
    Kokkos::deep_copy(scene.materials, mats_host);
  }

  if (!emissive_prim_ids.empty()) {
    scene.emissive_count = u32(emissive_prim_ids.size());
    scene.emissive_prim_ids = Kokkos::View<u32 *>("emissive_ids", scene.emissive_count);
    scene.emissive_prim_areas = Kokkos::View<f32 *>("emissive_areas", scene.emissive_count);

    auto eids_h = Kokkos::create_mirror_view(scene.emissive_prim_ids);
    auto eareas_h = Kokkos::create_mirror_view(scene.emissive_prim_areas);
    f32 total_area = 0.f;
    for (u32 i = 0; i < scene.emissive_count; ++i) {
      eids_h(i) = emissive_prim_ids[i];
      eareas_h(i) = emissive_prim_areas[i];
      total_area += emissive_prim_areas[i];
    }
    scene.total_emissive_area = total_area;
    Kokkos::deep_copy(scene.emissive_prim_ids, eids_h);
    Kokkos::deep_copy(scene.emissive_prim_areas, eareas_h);

    for (size_t mi = 0; mi < pbrt.meshes.size(); ++mi) {
      const auto &mesh = pbrt.meshes[mi];
      if (!mesh.is_emissive)
        continue;

      const size_t ntris = mesh.indices.size() / 3;
      u32 first_prim = emissive_prim_ids.empty() ? 0 : emissive_prim_ids[0];

      Mat4 xfm = mat4_from_pbrt_column_major(mesh.transform);
      Vec3 p0 = transform_point(xfm, {mesh.positions[mesh.indices[0]*3],
          mesh.positions[mesh.indices[0]*3+1], mesh.positions[mesh.indices[0]*3+2]});
      Vec3 p1 = transform_point(xfm, {mesh.positions[mesh.indices[1]*3],
          mesh.positions[mesh.indices[1]*3+1], mesh.positions[mesh.indices[1]*3+2]});
      Vec3 p3 = p0;
      if (mesh.indices.size() >= 6) {
        int i5 = mesh.indices[5];
        p3 = transform_point(xfm, {mesh.positions[i5*3], mesh.positions[i5*3+1], mesh.positions[i5*3+2]});
      }

      Light l{};
      l.type = LightType::Area;
      l.position = p0;
      l.edge1 = p1 - p0;
      l.edge2 = p3 - p0;
      l.area = total_area;
      l.color = {mesh.emission.x, mesh.emission.y, mesh.emission.z};
      l.intensity = 1.f;
      l.mesh_prim_begin = first_prim;
      l.mesh_prim_count = u32(ntris);
      Vec3 cr = cross(l.edge1, l.edge2);
      l.direction = length(cr) > 0.f ? cr * (1.f / length(cr)) : Vec3{0.f, -1.f, 0.f};
      lights_cpu.push_back(l);
    }
  }

  if (!lights_cpu.empty()) {
    scene.light_count = u32(lights_cpu.size());
    scene.lights = Kokkos::View<Light *>("lights", scene.light_count);
    auto lights_h = Kokkos::create_mirror_view(scene.lights);
    for (u32 i = 0; i < scene.light_count; ++i)
      lights_h(i) = lights_cpu[i];
    Kokkos::deep_copy(scene.lights, lights_h);
  }

  // pbrt Transform is world-to-camera; camera looks down +z in camera space
  Mat4 world_to_cam = mat4_from_pbrt_column_major(pbrt.camera.transform);
  Mat4 cam_to_world = world_to_cam.inverse();

  Vec3 cam_pos = cam_to_world.transform_point({0, 0, 0});
  Vec3 cam_dir = normalize(cam_to_world.transform_direction({0, 0, 1}));
  Vec3 cam_up = normalize(cam_to_world.transform_direction({0, 1, 0}));

  float aspect = float(pbrt.width) / float(pbrt.height);
  result.camera = Camera::make_perspective(
      cam_pos, cam_pos + cam_dir, cam_up, pbrt.camera.fov, aspect);

  return result;
}

}
