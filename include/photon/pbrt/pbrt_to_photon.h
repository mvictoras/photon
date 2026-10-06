#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include "photon/pt/camera.h"
#include "photon/pt/scene.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace photon::pbrt {

struct ConvertedScene {
  photon::pt::Scene scene;
  photon::pt::Camera camera;
  photon::pt::InstancedGeometry instanced_geometry;
};

ConvertedScene convert_pbrt_scene(const PbrtScene &pbrt, const std::string &base_dir = "");

// ---------------------------------------------------------------------------
// Sub-functions of convert_pbrt_scene, exposed so each concern can be unit
// tested in isolation with a small hand-built PbrtScene. convert_pbrt_scene
// is a thin orchestrator over these.
// ---------------------------------------------------------------------------

// Loaded-texture lookup: name -> atlas index, plus the atlas-ordered list.
struct PbrtTextureRefs {
  std::map<std::string, photon::pt::i32> name_to_id;
  std::vector<const PbrtTexture *> list;
};

// Load image data for every imagemap texture that has none yet (mutates the
// textures stored in the scene, hence the const_cast inside).
void load_pbrt_textures(const PbrtScene &pbrt, const std::string &base_dir);

// Collect textures that have image data, in deterministic atlas order.
PbrtTextureRefs collect_pbrt_textures(const PbrtScene &pbrt);

// Convert the scene's named PBRT materials into Material records, appending
// to out_materials. Returns the name -> material-id map. If the scene has no
// named materials, a single fallback material (id 0, "__default__") is added.
std::map<std::string, photon::pt::u32> convert_pbrt_materials(
    const PbrtScene &pbrt, const PbrtTextureRefs &textures,
    std::vector<photon::pt::Material> &out_materials);

struct PbrtMeshBuild {
  struct EmissiveMesh {
    photon::pt::u32 first_prim{0};
    photon::pt::u32 prim_count{0};
    photon::pt::f32 area{0.f};
  };

  photon::pt::TriangleMesh mesh;  // empty (0 triangles) for instanced-only scenes
  std::vector<photon::pt::u32> emissive_prim_ids;
  std::vector<photon::pt::f32> emissive_prim_areas;
  std::vector<EmissiveMesh> emissive_meshes;
  photon::pt::f32 total_emissive_area{0.f};  // sum of emissive_prim_areas
};

// Build the flat scene-level TriangleMesh from pbrt.meshes, applying each
// mesh's transform and deriving face normals where vertex normals are absent.
// Alpha-textured and emissive meshes append extra materials to `materials`
// (mutating both the vector and the id map). Meshes whose material is unknown
// map to material 0.
PbrtMeshBuild build_pbrt_triangle_mesh(
    const PbrtScene &pbrt, const PbrtTextureRefs &textures,
    std::vector<photon::pt::Material> &materials,
    std::map<std::string, photon::pt::u32> &mat_name_to_id);

// Derive area lights from the emissive primitives recorded by
// build_pbrt_triangle_mesh (one Light per emissive mesh).
std::vector<photon::pt::Light> derive_pbrt_area_lights(
    const PbrtScene &pbrt, const PbrtMeshBuild &built);

// Build the environment map (including the world<->texture rotation matrices
// from the scene's env transform). Returns nullopt when the scene has no env
// map or its image fails to load.
std::optional<photon::pt::EnvironmentMap> build_pbrt_env_map(
    const PbrtScene &pbrt, const std::string &base_dir);

// Derive the camera from LookAt coordinates, or from the camera transform's
// inverse when the scene has no LookAt.
photon::pt::Camera derive_pbrt_camera(const PbrtScene &pbrt);

// Build format-agnostic InstancedGeometry from object_defs/object_instances
// for backends that support IAS. Meshes are merged per object; the last
// sub-mesh's material wins when an object contains several.
photon::pt::InstancedGeometry build_pbrt_instanced_geometry(
    const PbrtScene &pbrt, const std::map<std::string, photon::pt::u32> &mat_name_to_id);

} // namespace photon::pbrt
