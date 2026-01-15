#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/math.h"
#include "photon/pt/math_aabb.h"

namespace photon::pt {

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

} // namespace photon::pt
