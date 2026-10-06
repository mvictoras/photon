#pragma once

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/environment_map.h"
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/math_mat4.h"
#include "photon/pt/texture.h"

#include <optional>
#include <vector>

namespace photon::pt {

// Format-agnostic mesh data for instanced acceleration structures.
// Used by backends that support two-level BVH (e.g. OptiX IAS).
// Scene converters (pbrt_to_photon, SceneFromAnari) populate these;
// backends consume them without knowing the source format.
struct ObjectMesh {
  std::vector<float> positions;   // xyz interleaved
  std::vector<int> indices;
  std::vector<float> normals;     // xyz interleaved (may be empty)
  std::vector<float> uvs;         // uv interleaved (may be empty)
  u32 material_id{0};
};

struct Instance {
  u32 object_id{0};
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct InstancedGeometry {
  std::vector<ObjectMesh> objects;   // one per unique object
  std::vector<Instance> instances;   // references objects by index
  bool empty() const { return objects.empty() || instances.empty(); }
};

struct Scene {
  TriangleMesh mesh;
  Bvh bvh;

  Kokkos::View<Material *> materials;
  u32 material_count{0};

  TextureAtlas textures;

  Kokkos::View<Light *> lights;
  u32 light_count{0};
  std::optional<EnvironmentMap> env_map;

  Kokkos::View<u32 *> emissive_prim_ids;
  Kokkos::View<f32 *> emissive_prim_areas;
  f32 total_emissive_area{0.f};
  u32 emissive_count{0};
};

// Marshal a format-agnostic ObjectMesh into a GPU TriangleMesh.
// Pure CPU-side data transform (Kokkos views only, no backend API) so it can
// be called and unit-tested from a CPU-only build without CUDA/OptiX.
// Positions, indices, normals and UVs are copied; material_ids are filled
// with the object's single material_id per triangle.
TriangleMesh object_mesh_to_triangle_mesh(const ObjectMesh &obj);

// CPU-side flattening fallback for backends without two-level BVH support:
// transforms each instance's referenced ObjectMesh into world space and
// appends it (together with the scene-level mesh) into a single flat
// TriangleMesh. Slower than a true IAS but never silently drops geometry.
TriangleMesh flatten_instanced_geometry(const Scene &scene,
                                        const InstancedGeometry &instanced);

} // namespace photon::pt
