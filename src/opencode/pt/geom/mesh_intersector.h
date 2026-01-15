#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/bvh/bvh.h"
#include "opencode/pt/geom/triangle_intersect.h"
#include "opencode/pt/ray.h"

namespace opencode::pt {

struct MeshHit
{
  f32 t{0.f};
  Vec3 p;
  Vec3 n;
  u32 prim_id{0};
  bool hit{false};
};

KOKKOS_FUNCTION inline MeshHit intersect_mesh_bvh(const TriangleMesh &mesh, const Bvh &bvh, const Ray &ray, f32 tmin,
    f32 tmax)
{
  MeshHit best;
  best.t = tmax;

  u32 stack[64];
  int sp = 0;
  stack[sp++] = bvh.root;

  while (sp > 0) {
    const u32 node_id = stack[--sp];
    const BvhNode node = bvh.nodes(node_id);

    if (!hit_aabb(node.bounds, ray, tmin, best.t))
      continue;

    if (node.is_leaf) {
      for (u32 i = 0; i < node.count; ++i) {
        const u32 prim_id = bvh.prim_ids(node.begin + i);
        const u32 i0 = mesh.indices(prim_id * 3 + 0);
        const u32 i1 = mesh.indices(prim_id * 3 + 1);
        const u32 i2 = mesh.indices(prim_id * 3 + 2);
        const Vec3 v0 = mesh.positions(i0);
        const Vec3 v1 = mesh.positions(i1);
        const Vec3 v2 = mesh.positions(i2);

        const TriHit h = intersect_triangle(ray, v0, v1, v2, tmin, best.t);
        if (!h.hit)
          continue;

        best.t = h.t;
        best.p = ray.at(h.t);
        Vec3 nn = normalize(cross(v1 - v0, v2 - v0));
        if (dot(nn, ray.dir) > 0.f)
          nn = nn * -1.f;
        best.n = nn;
        best.prim_id = prim_id;
        best.hit = true;
      }
    } else {
      if (sp + 2 <= 64) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }

  return best;
}

} // namespace opencode::pt
