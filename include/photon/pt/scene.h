#pragma once

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/environment_map.h"
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/math_mat4.h"
#include "photon/pt/texture.h"

#include <optional>

namespace photon::pt {

struct Instance {
  Mat4 transform;
  Mat4 inv_transform;
  u32 mesh_id{0};
  u32 material_id{0};
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

} // namespace photon::pt
