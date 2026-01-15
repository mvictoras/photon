#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/geom/triangle_mesh.h"
#include "opencode/pt/math.h"
#include "opencode/pt/math_aabb.h"

namespace opencode::pt {

struct BvhNode
{
  Aabb bounds;
  u32 left{0};
  u32 right{0};
  u32 begin{0};
  u32 count{0};
  u32 is_leaf{0};
};

struct Bvh
{
  Kokkos::View<BvhNode *> nodes;
  Kokkos::View<u32 *> prim_ids;
  u32 root{0};

  static Bvh build_cpu(const TriangleMesh &mesh);
};

} // namespace opencode::pt
