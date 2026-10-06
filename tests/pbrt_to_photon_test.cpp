// Unit tests for the convert_pbrt_scene sub-functions (each fed a small,
// hand-built PbrtScene — not a .pbrt file), plus one integration test over
// the convert_pbrt_scene seam itself.

#include "photon/pbrt/pbrt_to_photon.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace photon::pbrt;
using namespace photon::pt;

namespace {

bool approx(float a, float b, float eps = 1e-4f)
{
  return std::fabs(a - b) < eps;
}

bool approx3(const Vec3 &a, const Vec3 &b, float eps = 1e-4f)
{
  return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

// Two triangles forming a unit square in XY.
PbrtTriMesh make_square_mesh()
{
  PbrtTriMesh m;
  m.positions = {
      0.f, 0.f, 0.f,   1.f, 0.f, 0.f,   1.f, 1.f, 0.f,   // tri 0
      0.f, 0.f, 0.f,   1.f, 1.f, 0.f,   0.f, 1.f, 0.f}; // tri 1
  m.indices = {0, 1, 2, 3, 4, 5};
  m.uvs = {
      0.f, 0.f,   1.f, 0.f,   1.f, 1.f,
      0.f, 0.f,   1.f, 1.f,   0.f, 1.f};
  return m;
}

} // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {

    // ---- load_pbrt_textures -------------------------------------------
    {
      // Write a tiny 2x1 PFM for the imagemap texture.
      const char *path = "/tmp/photon_test_tex.pfm";
      {
        FILE *f = std::fopen(path, "wb");
        assert(f);
        std::fprintf(f, "PF\n2 1\n-1\n");
        float pixels[6] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
        std::fwrite(pixels, sizeof(float), 6, f);
        std::fclose(f);
      }

      PbrtScene pbrt;
      PbrtTexture imagemap;
      imagemap.class_type = "imagemap";
      imagemap.filename = path;
      pbrt.textures["img"] = imagemap;

      PbrtTexture constant;  // not an imagemap — must stay untouched
      constant.class_type = "constant";
      pbrt.textures["const"] = constant;

      PbrtTexture nofile;  // imagemap with a missing file — stays empty
      nofile.class_type = "imagemap";
      nofile.filename = "/tmp/definitely_missing_tex.pfm";
      pbrt.textures["nofile"] = nofile;

      load_pbrt_textures(pbrt, "");
      assert(pbrt.textures["img"].data.size() == 6);
      assert(pbrt.textures["img"].width == 2 && pbrt.textures["img"].height == 1);
      assert(pbrt.textures["const"].data.empty());
      assert(pbrt.textures["nofile"].data.empty());

      // Already-loaded textures are not reloaded (data preserved, no error).
      load_pbrt_textures(pbrt, "");
      assert(pbrt.textures["img"].data.size() == 6);
    }

    // ---- collect_pbrt_textures ----------------------------------------
    {
      PbrtScene pbrt;
      PbrtTexture loaded;
      loaded.data = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
      loaded.width = 2;
      loaded.height = 1;
      pbrt.textures["loaded"] = loaded;
      PbrtTexture unloaded;  // no image data — must be skipped
      pbrt.textures["unloaded"] = unloaded;

      auto refs = collect_pbrt_textures(pbrt);
      assert(refs.list.size() == 1);
      assert(refs.name_to_id.at("loaded") == 0);
      assert(refs.name_to_id.find("unloaded") == refs.name_to_id.end());
    }

    // ---- convert_pbrt_materials ---------------------------------------
    {
      PbrtScene pbrt;
      PbrtTextureRefs refs;
      PbrtMaterial diffuse;
      diffuse.name = "red";
      diffuse.type = "diffuse";
      diffuse.reflectance = {1.f, 0.f, 0.f};
      pbrt.named_materials["red"] = diffuse;

      std::vector<Material> mats;
      auto name_to_id = convert_pbrt_materials(pbrt, refs, mats);
      assert(mats.size() == 1);
      assert(name_to_id.at("red") == 0);
      assert(approx3(mats[0].base_color, {1.f, 0.f, 0.f}));
      assert(approx(mats[0].roughness, 1.f));

      // Unknown material types fall through to the reflectance/roughness path.
      PbrtMaterial weird;
      weird.name = "weird";
      weird.type = "whatever";
      weird.reflectance = {0.25f, 0.5f, 0.75f};
      weird.roughness = 0.3f;
      pbrt.named_materials["weird"] = weird;
      std::vector<Material> mats2;
      auto n2 = convert_pbrt_materials(pbrt, refs, mats2);
      assert(mats2.size() == 2);
      assert(approx3(mats2[1].base_color, {0.25f, 0.5f, 0.75f}));
      assert(approx(mats2[1].roughness, 0.3f));
    }

    // ---- convert_pbrt_materials: empty scene gets a fallback ----------
    {
      PbrtScene pbrt;
      PbrtTextureRefs refs;
      std::vector<Material> mats;
      auto n = convert_pbrt_materials(pbrt, refs, mats);
      assert(mats.size() == 1);
      assert(n.at("__default__") == 0);
      assert(approx3(mats[0].base_color, {0.5f, 0.5f, 0.5f}));
    }

    // ---- build_pbrt_triangle_mesh -------------------------------------
    {
      PbrtScene pbrt;
      PbrtTriMesh mesh = make_square_mesh();
      mesh.material_name = "red";
      // Translate +5 in x (column-major: translation in the 4th column).
      float tx[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 5,0,0,1};
      std::memcpy(mesh.transform, tx, sizeof(tx));
      pbrt.meshes.push_back(mesh);

      PbrtTextureRefs refs;
      std::vector<Material> materials;
      Material red{};
      red.base_color = {1.f, 0.f, 0.f};
      materials.push_back(red);
      std::map<std::string, u32> name_to_id{{"red", 0}};

      auto built = build_pbrt_triangle_mesh(pbrt, refs, materials, name_to_id);

      assert(built.mesh.triangle_count() == 2);
      assert(built.emissive_prim_ids.empty());

      auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, built.mesh.positions);
      auto nrm_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, built.mesh.normals);
      auto uv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, built.mesh.texcoords);
      auto mat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, built.mesh.material_ids);

      // Vertices transformed into world space.
      assert(approx3(pos_h(0), {5.f, 0.f, 0.f}));
      assert(approx3(pos_h(1), {6.f, 0.f, 0.f}));
      // No vertex normals supplied -> face normals are derived.
      assert(approx3(nrm_h(0), {0.f, 0.f, 1.f}));
      // UVs preserved, material resolved by name.
      assert(approx(uv_h(2).x, 1.f) && approx(uv_h(2).y, 1.f));
      assert(mat_h(0) == 0 && mat_h(1) == 0);
    }

    // ---- build_pbrt_triangle_mesh: emissive + alpha material appends ---
    {
      PbrtScene pbrt;
      PbrtTriMesh emitter = make_square_mesh();
      emitter.is_emissive = true;
      emitter.emission = {2.f, 2.f, 2.f};
      pbrt.meshes.push_back(emitter);

      PbrtTextureRefs refs;
      std::vector<Material> materials;
      Material white{};
      white.base_color = {0.7f, 0.7f, 0.7f};
      materials.push_back(white);
      std::map<std::string, u32> name_to_id{{"__default__", 0}};

      auto built = build_pbrt_triangle_mesh(pbrt, refs, materials, name_to_id);

      // One extra emissive material appended after the base material.
      assert(materials.size() == 2);
      assert(approx3(materials[1].emission, {2.f, 2.f, 2.f}));
      assert(built.emissive_prim_ids.size() == 2);
      assert(built.emissive_prim_areas.size() == 2);
      // Each triangle has area 0.5.
      assert(approx(built.emissive_prim_areas[0], 0.5f));
    }

    // ---- derive_pbrt_area_lights --------------------------------------
    {
      PbrtScene pbrt;
      PbrtTriMesh emitter = make_square_mesh();
      emitter.is_emissive = true;
      emitter.emission = {3.f, 1.f, 2.f};
      pbrt.meshes.push_back(emitter);

      PbrtMeshBuild built;
      built.emissive_prim_ids = {0, 1};
      built.emissive_prim_areas = {0.5f, 0.5f};
      built.total_emissive_area = 1.f;

      auto lights = derive_pbrt_area_lights(pbrt, built);
      assert(lights.size() == 1);
      const Light &l = lights[0];
      assert(l.type == LightType::Area);
      assert(approx3(l.position, {0.f, 0.f, 0.f}));
      assert(approx3(l.edge1, {1.f, 0.f, 0.f}));
      assert(approx3(l.edge2, {0.f, 1.f, 0.f}));
      assert(approx3(l.direction, {0.f, 0.f, 1.f}));
      assert(approx(l.area, 1.f));
      assert(l.mesh_prim_count == 2);
      assert(approx3(l.color, {3.f, 1.f, 2.f}));
    }

    // ---- build_pbrt_env_map -------------------------------------------
    {
      // Write a tiny 2x1 PFM next to the test so load_image can pick it up.
      const char *path = "/tmp/photon_test_env.pfm";
      {
        FILE *f = std::fopen(path, "wb");
        assert(f);
        std::fprintf(f, "PF\n2 1\n-1\n");
        float pixels[6] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
        std::fwrite(pixels, sizeof(float), 6, f);
        std::fclose(f);
      }

      PbrtScene pbrt;
      pbrt.has_env_map = true;
      pbrt.env_map_filename = path;
      pbrt.env_map_scale = 2.f;
      // Scale -1 in x (column-major diag(-1,1,1)); inverse is the same.
      float xfm[16] = {-1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      std::memcpy(pbrt.env_map_transform, xfm, sizeof(xfm));

      auto env = build_pbrt_env_map(pbrt, "");
      assert(env.has_value());
      assert(env->width == 2 && env->height == 1);
      // Rotation matrices extracted from the transform (and its inverse).
      assert(approx(env->tex_to_world[0], -1.f));
      assert(approx(env->world_to_tex[0], -1.f));
      assert(approx(env->tex_to_world[4], 1.f));
      assert(approx(env->world_to_tex[4], 1.f));
      assert(approx(env->tex_to_world[8], 1.f));

      auto pix_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, env->pixels);
      assert(approx(pix_h(0, 0).x, 2.f));  // data * env_map_scale

      // Missing file -> nullopt, not a crash.
      PbrtScene nofile;
      nofile.has_env_map = true;
      nofile.env_map_filename = "/tmp/definitely_missing_env.pfm";
      assert(!build_pbrt_env_map(nofile, ""));

      // No env map at all -> nullopt.
      PbrtScene none;
      assert(!build_pbrt_env_map(none, ""));
    }

    // ---- derive_pbrt_camera -------------------------------------------
    {
      PbrtScene pbrt;
      pbrt.camera.has_lookat = true;
      pbrt.camera.look_from = {0.f, 0.f, 5.f};
      pbrt.camera.look_at_pt = {0.f, 0.f, 0.f};
      pbrt.camera.look_up = {0.f, 1.f, 0.f};
      pbrt.width = 100;
      pbrt.height = 50;

      Camera cam = derive_pbrt_camera(pbrt);
      assert(approx3(cam.origin, {0.f, 0.f, 5.f}));

      // Transform-derived camera: identity world-to-cam means the camera
      // sits at the origin looking down +z.
      PbrtScene pbrt2;
      Camera cam2 = derive_pbrt_camera(pbrt2);
      assert(approx(cam2.origin.z, 0.f, 1e-5f));
    }

    // ---- build_pbrt_instanced_geometry --------------------------------
    {
      PbrtScene pbrt;

      PbrtTriMesh part_a;
      part_a.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
      part_a.indices = {0, 1, 2};
      part_a.material_name = "red";

      PbrtTriMesh part_b;
      part_b.positions = {0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 0.f, 1.f, 1.f};
      part_b.indices = {0, 1, 2};
      part_b.material_name = "white";

      pbrt.object_defs["cube"] = {part_a, part_b};
      pbrt.object_defs["orphan"] = {part_a};

      PbrtInstance inst_a;
      inst_a.object_name = "cube";
      PbrtInstance inst_b;
      inst_b.object_name = "cube";
      float tx[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 10,0,0,1};
      std::memcpy(inst_b.transform, tx, sizeof(tx));
      PbrtInstance inst_unknown;
      inst_unknown.object_name = "does_not_exist";

      pbrt.object_instances = {inst_a, inst_b, inst_unknown};

      std::map<std::string, u32> name_to_id{{"red", 0}, {"white", 1}};
      auto ig = build_pbrt_instanced_geometry(pbrt, name_to_id);

      assert(ig.objects.size() == 1);
      assert(ig.objects[0].positions.size() == 18);
      // Second sub-mesh's vertices are offset by the first's vertex count.
      assert(approx(ig.objects[0].positions[9], 0.f) &&
             approx(ig.objects[0].positions[11], 1.f));
      assert(ig.objects[0].indices[3] == 3);
      // Last sub-mesh's material wins.
      assert(ig.objects[0].material_id == 1);

      assert(ig.instances.size() == 2);  // unknown object name skipped
      assert(ig.instances[0].object_id == 0);
      assert(ig.instances[1].object_id == 0);
      assert(approx(ig.instances[1].transform[12], 10.f));
    }

    // ---- convert_pbrt_scene integration -------------------------------
    {
      PbrtScene pbrt;
      pbrt.width = 64;
      pbrt.height = 32;
      pbrt.camera.has_lookat = true;
      pbrt.camera.look_from = {0.f, 0.f, 5.f};
      pbrt.camera.look_at_pt = {0.f, 0.f, 0.f};
      pbrt.camera.look_up = {0.f, 1.f, 0.f};

      PbrtMaterial mat;
      mat.name = "red";
      mat.type = "diffuse";
      mat.reflectance = {1.f, 0.f, 0.f};
      pbrt.named_materials["red"] = mat;

      PbrtTriMesh mesh = make_square_mesh();
      mesh.material_name = "red";
      pbrt.meshes.push_back(mesh);

      auto converted = convert_pbrt_scene(pbrt);

      assert(converted.scene.mesh.triangle_count() == 2);
      assert(converted.scene.material_count == 1);
      assert(converted.scene.light_count == 0);
      assert(converted.instanced_geometry.empty());

      assert(approx3(converted.camera.origin, {0.f, 0.f, 5.f}));
    }

    // ---- convert_pbrt_scene integration: instanced-only scene ----------
    {
      PbrtScene pbrt;
      pbrt.width = 64;
      pbrt.height = 32;

      PbrtTriMesh part;
      part.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
      part.indices = {0, 1, 2};
      pbrt.object_defs["tri"] = {part};

      PbrtInstance inst;
      inst.object_name = "tri";
      pbrt.object_instances.push_back(inst);

      auto converted = convert_pbrt_scene(pbrt);

      // No scene-level triangles, but instancing data must survive.
      assert(converted.scene.mesh.triangle_count() == 0);
      assert(converted.instanced_geometry.objects.size() == 1);
      assert(converted.instanced_geometry.instances.size() == 1);
      // Materials still uploaded (fallback), camera still derived.
      assert(converted.scene.material_count == 1);
    }
  }
  Kokkos::finalize();
  return 0;
}
