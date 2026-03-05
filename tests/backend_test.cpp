#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/backend/kokkos_backend.h"
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

    auto backend = std::make_unique<KokkosBackend>();
    assert(std::strcmp(backend->name(), "kokkos") == 0);
    backend->build_accel(scene);

    RayBatch rays;
    rays.count = 2;
    rays.origins = Kokkos::View<Vec3 *>("org", 2);
    rays.directions = Kokkos::View<Vec3 *>("dir", 2);
    rays.tmin = Kokkos::View<f32 *>("tmin", 2);
    rays.tmax = Kokkos::View<f32 *>("tmax", 2);

    auto org_h = Kokkos::create_mirror_view(rays.origins);
    auto dir_h = Kokkos::create_mirror_view(rays.directions);
    auto tmin_h = Kokkos::create_mirror_view(rays.tmin);
    auto tmax_h = Kokkos::create_mirror_view(rays.tmax);

    org_h(0) = {0, 0, 2};
    dir_h(0) = {0, 0, -1};
    tmin_h(0) = 1e-3f;
    tmax_h(0) = 1e30f;
    org_h(1) = {0, 0, 2};
    dir_h(1) = {0, 0, 1};
    tmin_h(1) = 1e-3f;
    tmax_h(1) = 1e30f;

    Kokkos::deep_copy(rays.origins, org_h);
    Kokkos::deep_copy(rays.directions, dir_h);
    Kokkos::deep_copy(rays.tmin, tmin_h);
    Kokkos::deep_copy(rays.tmax, tmax_h);

    HitBatch hits;
    hits.count = 2;
    hits.hits = Kokkos::View<HitResult *>("hits", 2);
    backend->trace_closest(rays, hits);

    auto hits_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, hits.hits);
    assert(hits_h(0).hit == true);
    assert(hits_h(0).t > 0.f);
    assert(hits_h(1).hit == false);

    Kokkos::View<u32 *> occ("occ", 2);
    backend->trace_occluded(rays, occ);
    auto occ_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, occ);
    assert(occ_h(0) == 1u);
    assert(occ_h(1) == 0u);

    auto best = create_best_backend();
    assert(best != nullptr);
  }
  Kokkos::finalize();
  return 0;
}
