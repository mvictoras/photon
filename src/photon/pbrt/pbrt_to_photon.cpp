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
#include "photon/pt/environment_map.h"
#include "photon/pt/scene.h"

namespace photon::pbrt {

using namespace photon::pt;

namespace {

Vec3 sample_texture(const PbrtTexture &tex, float u, float v)
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

// Shared roughness rule for materials with (u,v) roughness pairs.
float pbrt_roughness(const PbrtMaterial &pmat)
{
  if (pmat.uroughness < 0.f)
    return pmat.roughness;
  const float v = pmat.vroughness >= 0.f ? pmat.vroughness : pmat.uroughness;
  return (pmat.uroughness + v) * 0.5f;
}

u32 lookup_material(const std::map<std::string, u32> &mat_name_to_id,
                    const std::string &name)
{
  auto it = mat_name_to_id.find(name);
  return it != mat_name_to_id.end() ? it->second : 0u;
}

} // anonymous namespace

void load_pbrt_textures(const PbrtScene &pbrt, const std::string &base_dir)
{
  auto &textures = const_cast<std::map<std::string, PbrtTexture> &>(pbrt.textures);
  for (auto &[name, tex] : textures) {
    if (tex.class_type == "imagemap" && !tex.filename.empty() && tex.data.empty()) {
      std::string path = base_dir + tex.filename;
      load_image(path, tex);
    }
  }
}

PbrtTextureRefs collect_pbrt_textures(const PbrtScene &pbrt)
{
  PbrtTextureRefs refs;
  for (const auto &[name, tex] : pbrt.textures) {
    if (!tex.data.empty() && tex.width > 0 && tex.height > 0) {
      refs.name_to_id[name] = i32(refs.list.size());
      refs.list.push_back(&tex);
    }
  }
  return refs;
}

std::map<std::string, u32> convert_pbrt_materials(
    const PbrtScene &pbrt, const PbrtTextureRefs &textures,
    std::vector<Material> &out_materials)
{
  std::map<std::string, u32> mat_name_to_id;

  for (const auto &[name, pmat] : pbrt.named_materials) {
    u32 id = u32(out_materials.size());
    mat_name_to_id[name] = id;

    Material m{};
    if (!pmat.reflectance_texture.empty()) {
      auto tit = textures.name_to_id.find(pmat.reflectance_texture);
      if (tit != textures.name_to_id.end())
        m.base_color_tex = tit->second;
    }

    if (pmat.type == "diffuse") {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.roughness = 1.f;
      m.metallic = 0.f;
    } else if (pmat.type == "conductor") {
      auto fresnel_f0 = [](float eta, float k) {
        float num = (eta - 1.f) * (eta - 1.f) + k * k;
        float den = (eta + 1.f) * (eta + 1.f) + k * k;
        return den > 0.f ? num / den : 0.f;
      };
      m.base_color = {
        fresnel_f0(pmat.eta.x, pmat.k.x),
        fresnel_f0(pmat.eta.y, pmat.k.y),
        fresnel_f0(pmat.eta.z, pmat.k.z)
      };
      m.metallic = 1.f;
      m.roughness = pbrt_roughness(pmat);
    } else if (pmat.type == "dielectric") {
      m.base_color = {1.f, 1.f, 1.f};
      m.ior = pmat.eta_scalar;
      m.transmission = 1.f;
      m.roughness = pbrt_roughness(pmat);
    } else if (pmat.type == "coateddiffuse") {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.metallic = 0.f;
      m.roughness = pbrt_roughness(pmat);
      m.clearcoat = 1.f;
      m.clearcoat_roughness = pbrt_roughness(pmat) * 0.5f;
    } else if (pmat.type == "diffusetransmission") {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.roughness = 1.f;
      m.metallic = 0.f;
      m.transmission = 0.5f;
    } else {
      m.base_color = {pmat.reflectance.x, pmat.reflectance.y, pmat.reflectance.z};
      m.roughness = pmat.roughness;
    }
    out_materials.push_back(m);
  }

  if (out_materials.empty()) {
    Material fallback{};
    fallback.base_color = {0.5f, 0.5f, 0.5f};
    fallback.roughness = 1.f;
    out_materials.push_back(fallback);
    mat_name_to_id["__default__"] = 0;
  }

  return mat_name_to_id;
}

PbrtMeshBuild build_pbrt_triangle_mesh(
    const PbrtScene &pbrt, const PbrtTextureRefs &textures,
    std::vector<Material> &materials, std::map<std::string, u32> &mat_name_to_id)
{
  PbrtMeshBuild out;

  uint64_t total_tris = 0;
  for (const auto &mesh : pbrt.meshes)
    total_tris += mesh.indices.size() / 3;

  if (total_tris == 0) {
    std::fprintf(stderr, "pbrt scene has no scene-level triangles (instanced-only scene)\n");
    return out;
  }

  const uint64_t total_verts = total_tris * 3;

  TriangleMesh tm;
  tm.positions = Kokkos::View<Vec3 *>("pos", total_verts);
  tm.indices = Kokkos::View<u32 *>("idx", total_verts);
  tm.material_ids = Kokkos::View<u32 *>("mat_id", total_tris);
  tm.albedo_per_prim = Kokkos::View<Vec3 *>("alb", total_tris);
  tm.normals = Kokkos::View<Vec3 *>("normals", total_verts);
  tm.texcoords = Kokkos::View<Vec2 *>("uv", total_verts);

  auto pos_h = Kokkos::create_mirror_view(tm.positions);
  auto idx_h = Kokkos::create_mirror_view(tm.indices);
  auto mat_id_h = Kokkos::create_mirror_view(tm.material_ids);
  auto alb_h = Kokkos::create_mirror_view(tm.albedo_per_prim);
  auto nrm_h = Kokkos::create_mirror_view(tm.normals);
  auto uv_h = Kokkos::create_mirror_view(tm.texcoords);

  uint64_t vert_off = 0;
  uint64_t tri_off = 0;

  for (const auto &mesh : pbrt.meshes) {
    // Per-mesh transform — decoded once, applied to every vertex/corner.
    const Mat4 xfm = mat4_from_column_major(mesh.transform);

    u32 mat_id = lookup_material(mat_name_to_id, mesh.material_name);

    const bool has_uvs = mesh.uvs.size() >= (mesh.positions.size() / 3) * 2;

    if (!mesh.alpha_texture.empty()) {
      auto ait = textures.name_to_id.find(mesh.alpha_texture);
      if (ait != textures.name_to_id.end()) {
        Material alpha_mat = materials[mat_id];
        alpha_mat.alpha_tex = ait->second;
        mat_id = u32(materials.size());
        materials.push_back(alpha_mat);
      }
    }

    if (mesh.is_emissive) {
      Material emit_mat{};
      emit_mat.base_color = {0.f, 0.f, 0.f};
      emit_mat.emission = {mesh.emission.x, mesh.emission.y, mesh.emission.z};
      emit_mat.emission_strength = 1.f;
      emit_mat.roughness = 1.f;
      mat_id = u32(materials.size());
      materials.push_back(emit_mat);
    }

    const size_t ntris = mesh.indices.size() / 3;
    const bool has_normals = mesh.normals.size() >= mesh.positions.size();

    for (size_t t = 0; t < ntris; ++t) {
      const int i0 = mesh.indices[t * 3 + 0];
      const int i1 = mesh.indices[t * 3 + 1];
      const int i2 = mesh.indices[t * 3 + 2];

      Vec3 p0 = xfm.transform_point({mesh.positions[i0*3], mesh.positions[i0*3+1], mesh.positions[i0*3+2]});
      Vec3 p1 = xfm.transform_point({mesh.positions[i1*3], mesh.positions[i1*3+1], mesh.positions[i1*3+2]});
      Vec3 p2 = xfm.transform_point({mesh.positions[i2*3], mesh.positions[i2*3+1], mesh.positions[i2*3+2]});

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
        nrm_h(vert_off + 0) = normalize(xfm.transform_direction(n0));
        nrm_h(vert_off + 1) = normalize(xfm.transform_direction(n1));
        nrm_h(vert_off + 2) = normalize(xfm.transform_direction(n2));
      } else {
        Vec3 face_n = normalize(cross(p1 - p0, p2 - p0));
        nrm_h(vert_off + 0) = face_n;
        nrm_h(vert_off + 1) = face_n;
        nrm_h(vert_off + 2) = face_n;
      }

      if (has_uvs) {
        uv_h(vert_off + 0) = {mesh.uvs[i0 * 2], mesh.uvs[i0 * 2 + 1]};
        uv_h(vert_off + 1) = {mesh.uvs[i1 * 2], mesh.uvs[i1 * 2 + 1]};
        uv_h(vert_off + 2) = {mesh.uvs[i2 * 2], mesh.uvs[i2 * 2 + 1]};
      }

      mat_id_h(tri_off) = mat_id;
      alb_h(tri_off) = materials[mat_id].base_color;

      if (mesh.is_emissive) {
        out.emissive_prim_ids.push_back(u32(tri_off));
        f32 area = 0.5f * length(cross(p1 - p0, p2 - p0));
        out.emissive_prim_areas.push_back(area);
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
  Kokkos::deep_copy(tm.texcoords, uv_h);

  out.mesh = tm;
  return out;
}

std::vector<Light> derive_pbrt_area_lights(
    const PbrtScene &pbrt, const std::vector<u32> &emissive_prim_ids,
    const std::vector<f32> &emissive_prim_areas, f32 total_emissive_area)
{
  std::vector<Light> lights;

  for (const auto &mesh : pbrt.meshes) {
    if (!mesh.is_emissive)
      continue;

    const size_t ntris = mesh.indices.size() / 3;
    u32 first_prim = emissive_prim_ids.empty() ? 0 : emissive_prim_ids[0];

    // Corner extraction: p0, p1 from the first triangle's first two indices,
    // p3 from its third index when present — forming the light quad edges.
    const Mat4 xfm = mat4_from_column_major(mesh.transform);
    auto corner = [&](int idx) {
      return xfm.transform_point({mesh.positions[idx*3], mesh.positions[idx*3+1], mesh.positions[idx*3+2]});
    };
    Vec3 p0 = corner(mesh.indices[0]);
    Vec3 p1 = corner(mesh.indices[1]);
    Vec3 p3 = mesh.indices.size() >= 6 ? corner(mesh.indices[5]) : p0;

    Light l{};
    l.type = LightType::Area;
    l.position = p0;
    l.edge1 = p1 - p0;
    l.edge2 = p3 - p0;
    l.area = total_emissive_area;
    l.color = {mesh.emission.x, mesh.emission.y, mesh.emission.z};
    l.intensity = 1.f;
    l.mesh_prim_begin = first_prim;
    l.mesh_prim_count = u32(ntris);
    Vec3 cr = cross(l.edge1, l.edge2);
    l.direction = length(cr) > 0.f ? cr * (1.f / length(cr)) : Vec3{0.f, -1.f, 0.f};
    lights.push_back(l);
  }

  return lights;
}

std::optional<EnvironmentMap> build_pbrt_env_map(const PbrtScene &pbrt,
                                                 const std::string &base_dir)
{
  if (!pbrt.has_env_map || pbrt.env_map_filename.empty())
    return std::nullopt;

  PbrtTexture env_tex;
  std::string env_path = base_dir + pbrt.env_map_filename;
  if (!load_image(env_path, env_tex))
    return std::nullopt;

  EnvironmentMap env;
  env.width = u32(env_tex.width);
  env.height = u32(env_tex.height);
  env.pixels = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("env_pixels", env.height, env.width);
  auto epix_h = Kokkos::create_mirror_view(env.pixels);
  for (u32 y = 0; y < env.height; ++y)
    for (u32 x = 0; x < env.width; ++x) {
      size_t si = (size_t(y) * env.width + x) * 3;
      epix_h(y, x) = {
        env_tex.data[si] * pbrt.env_map_scale,
        env_tex.data[si + 1] * pbrt.env_map_scale,
        env_tex.data[si + 2] * pbrt.env_map_scale};
    }
  Kokkos::deep_copy(env.pixels, epix_h);
  env.marginal_cdf = Kokkos::View<f32 *>("env_marginal", env.height + 1);
  env.conditional_cdf = Kokkos::View<f32 **>("env_conditional", env.height, env.width + 1);
  env.build_cdf();

  // Rotation matrices from the PBRT scene transform:
  // world_to_tex = inverse of the object-to-world env transform (3x3 upper-left),
  // tex_to_world = the object-to-world env transform itself.
  const Mat4 env_xfm = mat4_from_column_major(pbrt.env_map_transform);
  const Mat4 env_inv = env_xfm.inverse();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      env.world_to_tex[r * 3 + c] = env_inv.m[r][c];
      env.tex_to_world[r * 3 + c] = env_xfm.m[r][c];
    }
  std::fprintf(stderr, "  Env map rotation applied (transform from scene)\n");

  std::fprintf(stderr, "  Environment map: %s (%ux%u)\n", env_path.c_str(), env.width, env.height);
  return env;
}

Camera derive_pbrt_camera(const PbrtScene &pbrt)
{
  Vec3 cam_pos, cam_dir, cam_up;

  if (pbrt.camera.has_lookat) {
    // LookAt positions are in world space — Scale before Camera only affects
    // camera-internal handedness, not world-space eye/at/up coordinates.
    cam_pos = {pbrt.camera.look_from.x, pbrt.camera.look_from.y, pbrt.camera.look_from.z};
    Vec3 look_target = {pbrt.camera.look_at_pt.x, pbrt.camera.look_at_pt.y, pbrt.camera.look_at_pt.z};
    cam_dir = normalize(look_target - cam_pos);
    cam_up = normalize(Vec3{pbrt.camera.look_up.x, pbrt.camera.look_up.y, pbrt.camera.look_up.z});
  } else {
    Mat4 world_to_cam = mat4_from_column_major(pbrt.camera.transform);
    Mat4 cam_to_world = world_to_cam.inverse();
    cam_pos = cam_to_world.transform_point({0, 0, 0});
    cam_dir = normalize(cam_to_world.transform_direction({0, 0, 1}));
    cam_up = normalize(cam_to_world.transform_direction({0, 1, 0}));
  }

  float aspect = float(pbrt.width) / float(pbrt.height);
  Camera camera = Camera::make_perspective(
      cam_pos, cam_pos + cam_dir, cam_up, pbrt.camera.fov, aspect);
  if (pbrt.camera.lensradius > 0.f) {
    camera.lens_radius = pbrt.camera.lensradius;
    camera.focus_dist = pbrt.camera.focaldistance > 0.f
        ? pbrt.camera.focaldistance : length(cam_dir);
  }
  return camera;
}

InstancedGeometry build_pbrt_instanced_geometry(
    const PbrtScene &pbrt, const std::map<std::string, u32> &mat_name_to_id)
{
  InstancedGeometry ig;
  if (pbrt.object_defs.empty() || pbrt.object_instances.empty())
    return ig;

  std::map<std::string, u32> obj_name_to_idx;
  for (const auto &[name, meshes] : pbrt.object_defs) {
    u32 idx = u32(ig.objects.size());
    obj_name_to_idx[name] = idx;

    // Merge all sub-meshes of this object into one ObjectMesh.
    ObjectMesh obj;
    uint64_t voff = 0;
    for (const auto &m : meshes) {
      const uint64_t nv = m.positions.size() / 3;

      // Resolve material (last sub-mesh wins when multiple materials present)
      obj.material_id = lookup_material(mat_name_to_id, m.material_name);

      obj.positions.insert(obj.positions.end(),
          m.positions.begin(), m.positions.end());
      for (int idx_val : m.indices)
        obj.indices.push_back(idx_val + int(voff));
      if (m.normals.size() >= m.positions.size())
        obj.normals.insert(obj.normals.end(),
            m.normals.begin(), m.normals.end());
      if (!m.uvs.empty())
        obj.uvs.insert(obj.uvs.end(), m.uvs.begin(), m.uvs.end());

      voff += nv;
    }
    ig.objects.push_back(std::move(obj));
  }

  for (const auto &inst : pbrt.object_instances) {
    auto it = obj_name_to_idx.find(inst.object_name);
    if (it == obj_name_to_idx.end()) continue;

    Instance gi;
    gi.object_id = it->second;
    std::memcpy(gi.transform, inst.transform, sizeof(float) * 16);
    ig.instances.push_back(gi);
  }

  return ig;
}

ConvertedScene convert_pbrt_scene(const PbrtScene &pbrt, const std::string &base_dir)
{
  ConvertedScene result;
  Scene &scene = result.scene;

  load_pbrt_textures(pbrt, base_dir);
  const PbrtTextureRefs textures = collect_pbrt_textures(pbrt);

  std::vector<Material> materials_cpu;
  std::map<std::string, u32> mat_name_to_id =
      convert_pbrt_materials(pbrt, textures, materials_cpu);

  const PbrtMeshBuild built =
      build_pbrt_triangle_mesh(pbrt, textures, materials_cpu, mat_name_to_id);

  if (built.mesh.triangle_count() > 0) {
    scene.mesh = built.mesh;
    scene.bvh = Bvh::build_cpu(built.mesh);
  }

  scene.material_count = u32(materials_cpu.size());
  {
    Kokkos::View<Material *, Kokkos::HostSpace> mats_host("mats_host", scene.material_count);
    for (u32 i = 0; i < scene.material_count; ++i)
      mats_host(i) = materials_cpu[i];
    scene.materials = Kokkos::View<Material *>("materials", scene.material_count);
    Kokkos::deep_copy(scene.materials, mats_host);
  }

  if (!textures.list.empty()) {
    scene.textures.count = u32(textures.list.size());

    size_t total_pixels = 0;
    for (const auto *src : textures.list)
      total_pixels += size_t(src->width) * size_t(src->height);

    scene.textures.pixels = Kokkos::View<Vec3 *>("tex_pixels", total_pixels);
    scene.textures.infos = Kokkos::View<TextureInfo *>("tex_infos", scene.textures.count);

    auto pix_h = Kokkos::create_mirror_view(scene.textures.pixels);
    auto info_h = Kokkos::create_mirror_view(scene.textures.infos);

    u32 pixel_offset = 0;
    for (u32 ti = 0; ti < scene.textures.count; ++ti) {
      const PbrtTexture *src = textures.list[ti];
      info_h(ti).offset = pixel_offset;
      info_h(ti).width = u32(src->width);
      info_h(ti).height = u32(src->height);

      for (u32 y = 0; y < u32(src->height); ++y)
        for (u32 x = 0; x < u32(src->width); ++x) {
          size_t si = (size_t(y) * src->width + x) * 3;
          pix_h(pixel_offset + y * u32(src->width) + x) =
              {src->data[si], src->data[si + 1], src->data[si + 2]};
        }
      pixel_offset += u32(src->width) * u32(src->height);
    }

    Kokkos::deep_copy(scene.textures.pixels, pix_h);
    Kokkos::deep_copy(scene.textures.infos, info_h);
    std::fprintf(stderr, "  Uploaded %u textures (%zu pixels, %.1f MB) to GPU\n",
        scene.textures.count, total_pixels, float(total_pixels * sizeof(Vec3)) / (1024.f * 1024.f));
  }

  if (!built.emissive_prim_ids.empty()) {
    scene.emissive_count = u32(built.emissive_prim_ids.size());
    scene.emissive_prim_ids = Kokkos::View<u32 *>("emissive_ids", scene.emissive_count);
    scene.emissive_prim_areas = Kokkos::View<f32 *>("emissive_areas", scene.emissive_count);

    auto eids_h = Kokkos::create_mirror_view(scene.emissive_prim_ids);
    auto eareas_h = Kokkos::create_mirror_view(scene.emissive_prim_areas);
    f32 total_area = 0.f;
    for (u32 i = 0; i < scene.emissive_count; ++i) {
      eids_h(i) = built.emissive_prim_ids[i];
      eareas_h(i) = built.emissive_prim_areas[i];
      total_area += built.emissive_prim_areas[i];
    }
    scene.total_emissive_area = total_area;
    Kokkos::deep_copy(scene.emissive_prim_ids, eids_h);
    Kokkos::deep_copy(scene.emissive_prim_areas, eareas_h);
  }

  const std::vector<Light> lights_cpu = derive_pbrt_area_lights(
      pbrt, built.emissive_prim_ids, built.emissive_prim_areas,
      scene.total_emissive_area);

  if (!lights_cpu.empty()) {
    scene.light_count = u32(lights_cpu.size());
    scene.lights = Kokkos::View<Light *>("lights", scene.light_count);
    auto lights_h = Kokkos::create_mirror_view(scene.lights);
    for (u32 i = 0; i < scene.light_count; ++i)
      lights_h(i) = lights_cpu[i];
    Kokkos::deep_copy(scene.lights, lights_h);
  }

  if (auto env = build_pbrt_env_map(pbrt, base_dir))
    scene.env_map = std::move(*env);

  result.camera = derive_pbrt_camera(pbrt);
  result.instanced_geometry = build_pbrt_instanced_geometry(pbrt, mat_name_to_id);

  return result;
}

} // namespace photon::pbrt
