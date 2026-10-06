// Verifies the instancing seam on RayBackend:
//  - backends without IAS support report supports_instancing() == false
//  - the default build_accel_instanced flattens instances into the flat
//    path instead of silently dropping them
//  - rays hit geometry that only exists inside InstancedGeometry

#include "photon/pt/backend/kokkos_backend.h"
#include "photon/pt/scene.h"
#include "photon/pt/scene/builder.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>
#include <cstring>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    auto scene = SceneBuilder::make_two_quads();

    // One object: a single triangle in the XY plane at the origin.
    InstancedGeometry ig;
    ObjectMesh obj;
    obj.positions = {0.f, 0.f, 0.f,   1.f, 0.f, 0.f,   0.f, 1.f, 0.f};
    obj.indices = {0, 1, 2};
    obj.normals = {0.f, 0.f, 1.f,   0.f, 0.f, 1.f,   0.f, 0.f, 1.f};
    obj.uvs = {0.f, 0.f,   1.f, 0.f,   0.f, 1.f};
    obj.material_id = 0;
    ig.objects.push_back(obj);

    Instance inst_a{};  // identity — triangle stays at origin
    ig.instances.push_back(inst_a);

    Instance inst_b{};  // translate +10 in x (column-major: translation in col 3)
    inst_b.object_id = 0;
    float tx[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  10,0,0,1};
    std::memcpy(inst_b.transform, tx, sizeof(tx));
    ig.instances.push_back(inst_b);

    // Direct flattening: scene mesh (4 tris) + 2 instance triangles.
    TriangleMesh flat = flatten_instanced_geometry(scene, ig);
    assert(flat.triangle_count() == 6);
    assert(flat.has_texcoords());

    auto flat_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flat.positions);
    // First instance triangle lands at the origin (vertices 12..14).
    assert(std::abs(flat_pos(12).x - 0.f) < 1e-5f);
    // Second instance is translated by +10 in x (vertices 15..17).
    assert(std::abs(flat_pos(15).x - 10.f) < 1e-5f);

    // Through the backend interface: KokkosBackend has no IAS support, so
    // build_accel_instanced must flatten (rendering, not dropping).
    auto backend = std::make_unique<KokkosBackend>();
    assert(backend->supports_instancing() == false);
    backend->build_accel_instanced(scene, ig);

    RayBatch rays;
    rays.count = 3;
    rays.origins = Kokkos::View<Vec3 *>("org", 3);
    rays.directions = Kokkos::View<Vec3 *>("dir", 3);
    rays.tmin = Kokkos::View<f32 *>("tmin", 3);
    rays.tmax = Kokkos::View<f32 *>("tmax", 3);

    auto org_h = Kokkos::create_mirror_view(rays.origins);
    auto dir_h = Kokkos::create_mirror_view(rays.directions);
    auto tmin_h = Kokkos::create_mirror_view(rays.tmin);
    auto tmax_h = Kokkos::create_mirror_view(rays.tmax);

    // Ray 0: straight at the identity-instance triangle (z=0 plane).
    org_h(0) = {0.25f, 0.25f, 5.f};
    dir_h(0) = {0.f, 0.f, -1.f};
    // Ray 1: straight at the translated instance triangle (x+10).
    org_h(1) = {10.25f, 0.25f, 5.f};
    dir_h(1) = {0.f, 0.f, -1.f};
    // Ray 2: misses both instances (y<0), hits the scene quad at z=-2.5.
    org_h(2) = {0.f, -0.5f, 2.f};
    dir_h(2) = {0.f, 0.f, -1.f};

    for (u32 i = 0; i < 3; ++i) {
      tmin_h(i) = 1e-3f;
      tmax_h(i) = 1e30f;
    }

    Kokkos::deep_copy(rays.origins, org_h);
    Kokkos::deep_copy(rays.directions, dir_h);
    Kokkos::deep_copy(rays.tmin, tmin_h);
    Kokkos::deep_copy(rays.tmax, tmax_h);

    HitBatch hits;
    hits.count = 3;
    hits.hits = Kokkos::View<HitResult *>("hits", 3);
    backend->trace_closest(rays, hits);

    auto hits_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hits.hits);

    // The regression: instanced geometry must be present. Before the fix,
    // the silent default dropped it and rays 0/1 missed entirely.
    assert(hits_h(0).hit);
    assert(std::abs(hits_h(0).t - 5.f) < 1e-3f);
    assert(hits_h(1).hit);
    assert(std::abs(hits_h(1).t - 5.f) < 1e-3f);
    assert(hits_h(2).hit);
    assert(std::abs(hits_h(2).t - 4.5f) < 1e-3f);

    // Empty instanced geometry falls through to a plain flat build.
    auto backend2 = std::make_unique<KokkosBackend>();
    InstancedGeometry empty_ig;
    backend2->build_accel_instanced(scene, empty_ig);
    RayBatch rays2;
    rays2.count = 1;
    rays2.origins = Kokkos::View<Vec3 *>("org2", 1);
    rays2.directions = Kokkos::View<Vec3 *>("dir2", 1);
    rays2.tmin = Kokkos::View<f32 *>("tmin2", 1);
    rays2.tmax = Kokkos::View<f32 *>("tmax2", 1);
    auto org2_h = Kokkos::create_mirror_view(rays2.origins);
    auto dir2_h = Kokkos::create_mirror_view(rays2.directions);
    auto tmin2_h = Kokkos::create_mirror_view(rays2.tmin);
    auto tmax2_h = Kokkos::create_mirror_view(rays2.tmax);
    org2_h(0) = {0.f, 0.f, 2.f};
    dir2_h(0) = {0.f, 0.f, -1.f};
    tmin2_h(0) = 1e-3f;
    tmax2_h(0) = 1e30f;
    Kokkos::deep_copy(rays2.origins, org2_h);
    Kokkos::deep_copy(rays2.directions, dir2_h);
    Kokkos::deep_copy(rays2.tmin, tmin2_h);
    Kokkos::deep_copy(rays2.tmax, tmax2_h);
    HitBatch hits2;
    hits2.count = 1;
    hits2.hits = Kokkos::View<HitResult *>("hits2", 1);
    backend2->trace_closest(rays2, hits2);
    auto hits2_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hits2.hits);
    assert(hits2_h(0).hit);
    assert(std::abs(hits2_h(0).t - 4.5f) < 1e-3f);
  }
  Kokkos::finalize();
  return 0;
}
