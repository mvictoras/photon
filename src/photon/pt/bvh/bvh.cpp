#include "photon/pt/bvh/bvh.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "photon/pt/geom/triangle_intersect.h"

namespace photon::pt {

namespace {

struct PrimRef
{
  u32 id;
  Aabb bounds;
  Vec3 centroid;
};

Aabb tri_bounds(const TriangleMesh &mesh, u32 tri)
{
  const u32 i0 = mesh.indices(tri * 3 + 0);
  const u32 i1 = mesh.indices(tri * 3 + 1);
  const u32 i2 = mesh.indices(tri * 3 + 2);

  const Vec3 v0 = mesh.positions(i0);
  const Vec3 v1 = mesh.positions(i1);
  const Vec3 v2 = mesh.positions(i2);

  Aabb b;
  b.extend(v0);
  b.extend(v1);
  b.extend(v2);
  return b;
}

Vec3 tri_centroid(const TriangleMesh &mesh, u32 tri)
{
  const u32 i0 = mesh.indices(tri * 3 + 0);
  const u32 i1 = mesh.indices(tri * 3 + 1);
  const u32 i2 = mesh.indices(tri * 3 + 2);
  const Vec3 v0 = mesh.positions(i0);
  const Vec3 v1 = mesh.positions(i1);
  const Vec3 v2 = mesh.positions(i2);
  return (v0 + v1 + v2) * (1.f / 3.f);
}

u32 build_node(std::vector<BvhNode> &nodes, std::vector<PrimRef> &prims, std::vector<u32> &prim_ids, u32 begin,
    u32 end)
{
  const u32 node_id = u32(nodes.size());
  nodes.push_back({});

  Aabb b;
  Aabb cbox;
  for (u32 i = begin; i < end; ++i) {
    b.extend(prims[i].bounds);
    cbox.extend(prims[i].centroid);
  }

  BvhNode &n = nodes[node_id];
  n.bounds = b;
  const u32 prim_begin = u32(prim_ids.size());
  for (u32 i = begin; i < end; ++i)
    prim_ids.push_back(prims[i].id);

  n.begin = prim_begin;
  n.count = end - begin;

  constexpr u32 leaf_size = 4;
  if (n.count <= leaf_size) {
    n.is_leaf = 1;
    n.left = 0;
    n.right = 0;
    return node_id;
  }

  prim_ids.resize(prim_begin);
  n.begin = begin;

  const Vec3 diag = cbox.hi - cbox.lo;
  int axis = 0;
  if (diag.y > diag.x)
    axis = 1;
  if ((axis == 0 ? diag.z : (axis == 1 ? diag.z : diag.y)) > (axis == 0 ? diag.x : diag.y))
    axis = 2;

  const auto key = [axis](const PrimRef &p) {
    return axis == 0 ? p.centroid.x : (axis == 1 ? p.centroid.y : p.centroid.z);
  };

  const u32 mid = begin + (end - begin) / 2;
  std::nth_element(prims.begin() + begin, prims.begin() + mid, prims.begin() + end,
      [&](const PrimRef &a, const PrimRef &b) { return key(a) < key(b); });


  n.left = build_node(nodes, prims, prim_ids, begin, mid);
  n.right = build_node(nodes, prims, prim_ids, mid, end);
  n.is_leaf = 0;
  return node_id;
}

} // namespace

Bvh Bvh::build_cpu(const TriangleMesh &mesh)
{
  const u32 ntris = mesh.triangle_count();

  auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions);
  auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.indices);

  TriangleMesh host_mesh{pos_h, idx_h, {}};

  std::vector<PrimRef> prims(ntris);
  for (u32 i = 0; i < ntris; ++i) {
    prims[i].id = i;
    prims[i].bounds = tri_bounds(host_mesh, i);
    prims[i].centroid = tri_centroid(host_mesh, i);
  }


  std::vector<u32> prim_ids;
  prim_ids.reserve(ntris);

  std::vector<BvhNode> nodes;
  nodes.reserve(ntris * 2);

  const u32 root = build_node(nodes, prims, prim_ids, 0, ntris);

  Bvh out;
  out.root = root;
  out.nodes = Kokkos::View<BvhNode *>("bvh_nodes", nodes.size());
  out.prim_ids = Kokkos::View<u32 *>("bvh_prim_ids", prim_ids.size());

  auto nodes_h = Kokkos::create_mirror_view(out.nodes);
  auto ids_h = Kokkos::create_mirror_view(out.prim_ids);

  for (size_t i = 0; i < nodes.size(); ++i)
    nodes_h(i) = nodes[i];
  for (size_t i = 0; i < prim_ids.size(); ++i)
    ids_h(i) = prim_ids[i];

  Kokkos::deep_copy(out.nodes, nodes_h);
  Kokkos::deep_copy(out.prim_ids, ids_h);

  return out;
}

} // namespace photon::pt
