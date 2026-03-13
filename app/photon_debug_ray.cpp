#include <Kokkos_Core.hpp>

#include <cstdio>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/camera.h"
#include "photon/pt/geom/mesh_intersector.h"
#include "photon/pt/geom/triangle_intersect.h"
#include "photon/pt/scene.h"
#include "photon/pt/bvh/bvh_validate.h"

namespace {

photon::pt::Scene make_quad_scene()
{
  using namespace photon::pt;

  TriangleMesh mesh;
  mesh.positions = Kokkos::View<Vec3 *>("pos", 4);
  mesh.indices = Kokkos::View<u32 *>("idx", 6);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);

  pos_h(0) = {-1.f, -1.f, -2.5f};
  pos_h(1) = {+1.f, -1.f, -2.5f};
  pos_h(2) = {+1.f, +1.f, -2.5f};
  pos_h(3) = {-1.f, +1.f, -2.5f};

  idx_h(0) = 0;
  idx_h(1) = 1;
  idx_h(2) = 2;
  idx_h(3) = 0;
  idx_h(4) = 2;
  idx_h(5) = 3;

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);

  Scene s;
  s.mesh = mesh;
  s.bvh = Bvh::build_cpu(mesh);
  return s;
}

bool intersect_mesh_direct(const photon::pt::TriangleMesh &mesh, const photon::pt::Ray &ray)
{
  using namespace photon::pt;
  auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions);
  auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.indices);

  const u32 n = u32(idx_h.extent(0) / 3);
  for (u32 tri = 0; tri < n; ++tri) {
    const u32 i0 = idx_h(tri * 3 + 0);
    const u32 i1 = idx_h(tri * 3 + 1);
    const u32 i2 = idx_h(tri * 3 + 2);

    const Vec3 v0 = pos_h(i0);
    const Vec3 v1 = pos_h(i1);
    const Vec3 v2 = pos_h(i2);

    const TriHit h = intersect_triangle(ray, v0, v1, v2, 1e-3f, 1e30f);
    if (h.hit)
      return true;
  }

  return false;
}

} // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    const u32 w = 512;
    const u32 h = 512;
    const f32 aspect = f32(w) / f32(h);
    const Camera cam = Camera::make_pinhole({3.f, 3.f, 2.f}, {0.f, 0.f, -2.5f}, {0.f, 1.f, 0.f}, 20.f, aspect);

    const f32 u = 0.5f;
    const f32 v = 0.5f;
    const Ray r = cam.ray(u, v);

    auto scene = make_quad_scene();

    const bool bvh_ok = validate_bvh_host(scene.bvh);

    const bool direct_hit = intersect_mesh_direct(scene.mesh, r);
    const auto nodes_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scene.bvh.nodes);
    const auto &node0 = nodes_h(scene.bvh.root);
    const bool root_aabb_hit = hit_aabb(node0.bounds, r, 1e-3f, 1e30f);

    std::printf("root_aabb lo=(%g,%g,%g) hi=(%g,%g,%g)\n", node0.bounds.lo.x, node0.bounds.lo.y, node0.bounds.lo.z,
        node0.bounds.hi.x, node0.bounds.hi.y, node0.bounds.hi.z);

    const MeshHit bvh_hit = intersect_mesh_bvh(scene.mesh, scene.bvh, r, 1e-3f, 1e30f);

    std::printf("ray org=(%g,%g,%g) dir=(%g,%g,%g)\n", r.org.x, r.org.y, r.org.z, r.dir.x, r.dir.y, r.dir.z);
    std::printf("bvh_ok=%d\n", int(bvh_ok));
    std::printf("direct_hit=%d\n", int(direct_hit));
    std::printf("root_aabb_hit=%d\n", int(root_aabb_hit));
    std::printf("bvh_hit=%d t=%g\n", int(bvh_hit.hit), bvh_hit.t);
  }
  Kokkos::finalize();
  return 0;
}
