#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_intersect.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/ray.h"

namespace photon::pt {

struct MeshHit
{
  f32 t{0.f};
  Vec3 p;
  Vec3 n;
  Vec3 shading_normal;
  Vec2 uv{0.f, 0.f};
  Vec3 interpolated_color{0.f, 0.f, 0.f};
  u32 prim_id{0};
  u32 geom_id{0};
  u32 material_id{0};
  bool hit{false};
  bool has_interpolated_color{false};
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
        const Vec3 nn = normalize(cross(v1 - v0, v2 - v0));
        best.n = nn;
        best.shading_normal = nn;
        best.uv = Vec2{0.f, 0.f};
        best.prim_id = prim_id;
        best.geom_id = 0;
        best.material_id = 0;

        const f32 w0 = 1.f - h.u - h.v;
        const f32 w1 = h.u;
        const f32 w2 = h.v;

        if (mesh.has_normals()) {
          const Vec3 n0 = mesh.normals(i0);
          const Vec3 n1 = mesh.normals(i1);
          const Vec3 n2 = mesh.normals(i2);
          best.shading_normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
          if (dot(best.shading_normal, nn) < 0.f)
            best.shading_normal = best.shading_normal * -1.f;
        }

        if (mesh.has_texcoords()) {
          const Vec2 uv0 = mesh.texcoords(i0);
          const Vec2 uv1 = mesh.texcoords(i1);
          const Vec2 uv2 = mesh.texcoords(i2);
          best.uv = Vec2{uv0.x * w0 + uv1.x * w1 + uv2.x * w2, uv0.y * w0 + uv1.y * w1 + uv2.y * w2};
        }

        if (mesh.has_vertex_colors()) {
          const Vec3 vc0 = mesh.vertex_colors(i0);
          const Vec3 vc1 = mesh.vertex_colors(i1);
          const Vec3 vc2 = mesh.vertex_colors(i2);
          best.interpolated_color = vc0 * w0 + vc1 * w1 + vc2 * w2;
          best.has_interpolated_color = true;
        }

        if (mesh.has_material_ids()) {
          best.material_id = mesh.material_ids(prim_id);
        }

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

} // namespace photon::pt
