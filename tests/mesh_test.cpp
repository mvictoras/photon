#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/mesh_intersector.h"
#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/ray.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>

namespace {

using photon::pt::Bvh;
using photon::pt::MeshHit;
using photon::pt::Ray;
using photon::pt::TriangleMesh;
using photon::pt::Vec2;
using photon::pt::Vec3;
using photon::pt::f32;
using photon::pt::u32;

inline bool near(f32 a, f32 b, f32 eps = 1e-4f) { return std::fabs(a - b) <= eps; }
inline bool near(Vec2 a, Vec2 b, f32 eps = 1e-4f) { return near(a.x, b.x, eps) && near(a.y, b.y, eps); }
inline bool near(Vec3 a, Vec3 b, f32 eps = 1e-4f) { return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps); }

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    TriangleMesh mesh;
    mesh.positions = Kokkos::View<Vec3 *>("pos", 3);
    mesh.indices = Kokkos::View<u32 *>("idx", 3);
    mesh.albedo_per_prim = Kokkos::View<Vec3 *>("alb", 1);
    mesh.normals = Kokkos::View<Vec3 *>("n", 3);
    mesh.texcoords = Kokkos::View<Vec2 *>("uv", 3);
    mesh.material_ids = Kokkos::View<u32 *>("mat", 1);

    auto pos_h = Kokkos::create_mirror_view(mesh.positions);
    auto idx_h = Kokkos::create_mirror_view(mesh.indices);
    auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);
    auto n_h = Kokkos::create_mirror_view(mesh.normals);
    auto uv_h = Kokkos::create_mirror_view(mesh.texcoords);
    auto mat_h = Kokkos::create_mirror_view(mesh.material_ids);

    pos_h(0) = Vec3{0.f, 0.f, 0.f};
    pos_h(1) = Vec3{1.f, 0.f, 0.f};
    pos_h(2) = Vec3{0.f, 1.f, 0.f};
    idx_h(0) = 0; idx_h(1) = 1; idx_h(2) = 2;
    alb_h(0) = Vec3{1.f, 0.f, 0.f};
    n_h(0) = Vec3{0.f, 0.f, 1.f};
    n_h(1) = Vec3{0.f, 1.f, 0.f};
    n_h(2) = Vec3{1.f, 0.f, 0.f};
    uv_h(0) = Vec2{0.f, 0.f};
    uv_h(1) = Vec2{1.f, 0.f};
    uv_h(2) = Vec2{0.f, 1.f};
    mat_h(0) = 7u;

    Kokkos::deep_copy(mesh.positions, pos_h);
    Kokkos::deep_copy(mesh.indices, idx_h);
    Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);
    Kokkos::deep_copy(mesh.normals, n_h);
    Kokkos::deep_copy(mesh.texcoords, uv_h);
    Kokkos::deep_copy(mesh.material_ids, mat_h);

    Bvh bvh = Bvh::build_cpu(mesh);

    Kokkos::View<MeshHit *> results("results", 2);
    Ray ray{Vec3{0.25f, 0.25f, 1.f}, Vec3{0.f, 0.f, -1.f}};

    Kokkos::parallel_for("mesh_test", 1, KOKKOS_LAMBDA(int) {
      results(0) = photon::pt::intersect_mesh_bvh(mesh, bvh, ray, 1e-3f, 1e30f);
    });

    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    MeshHit hit = res_h(0);

    assert(hit.hit);
    assert(near(hit.p, Vec3{0.25f, 0.25f, 0.f}));
    assert(near(hit.uv, Vec2{0.25f, 0.25f}));
    assert(hit.material_id == 7u);

    Vec3 expected_sn = photon::pt::normalize(
        Vec3{0.f, 0.f, 1.f} * 0.5f + Vec3{0.f, 1.f, 0.f} * 0.25f + Vec3{1.f, 0.f, 0.f} * 0.25f);
    assert(near(hit.shading_normal, expected_sn));

    TriangleMesh mesh2;
    mesh2.positions = mesh.positions;
    mesh2.indices = mesh.indices;
    mesh2.albedo_per_prim = mesh.albedo_per_prim;

    Bvh bvh2 = Bvh::build_cpu(mesh2);

    Kokkos::parallel_for("mesh_test2", 1, KOKKOS_LAMBDA(int) {
      results(0) = photon::pt::intersect_mesh_bvh(mesh2, bvh2, ray, 1e-3f, 1e30f);
    });

    auto res2_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    MeshHit hit2 = res2_h(0);

    assert(hit2.hit);
    assert(near(hit2.shading_normal, hit2.n));
    assert(near(hit2.uv, Vec2{0.f, 0.f}));
  }
  Kokkos::finalize();
  return 0;
}
