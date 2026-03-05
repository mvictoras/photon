#include "photon/pt/volume.h"

#include <Kokkos_Core.hpp>

#include <cassert>

namespace {

using photon::pt::f32;
using photon::pt::Rng;
using photon::pt::Ray;
using photon::pt::u32;
using photon::pt::Vec3;
using photon::pt::VolumeEvent;
using photon::pt::VolumeGrid;

KOKKOS_FUNCTION inline bool near(f32 a, f32 b, f32 eps = 1e-5f)
{
  return Kokkos::fabs(a - b) <= eps;
}

KOKKOS_FUNCTION inline bool inside(const Vec3 &p, const Vec3 &lo, const Vec3 &hi)
{
  return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z;
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    VolumeGrid vol;
    vol.bounds_lo = Vec3{-1.f, -1.f, -1.f};
    vol.bounds_hi = Vec3{+1.f, +1.f, +1.f};
    vol.max_density = 1.f;
    vol.sigma_s = Vec3{1.f, 1.f, 1.f};
    vol.sigma_a = Vec3{0.1f, 0.1f, 0.1f};
    vol.g = 0.2f;

    vol.density = Kokkos::View<f32 ***, Kokkos::LayoutRight>("density", 4, 4, 4);
    auto host = Kokkos::create_mirror_view(vol.density);
    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
          host(z, y, x) = 1.f;
    Kokkos::deep_copy(vol.density, host);

    {
      Ray r{Vec3{-2.f, 0.f, 0.f}, Vec3{1.f, 0.f, 0.f}};
      f32 t0 = 0.f, t1 = 0.f;
      const bool hit = vol.intersect_bounds(r, t0, t1);
      assert(hit);
      assert(near(t0, 1.f));
      assert(near(t1, 3.f));
    }

    {
      Rng rng(123u);
      const Vec3 wo = photon::pt::normalize(Vec3{0.1f, 0.2f, 1.f});
      const Vec3 wi = photon::pt::sample_hg(wo, 0.3f, rng);
      assert(near(photon::pt::dot(wi, wi), 1.f, 1e-3f));
      const f32 p = photon::pt::pdf_hg(photon::pt::dot(wo, wi), 0.3f);
      assert(p > 0.f);
    }

    {
      const f32 p = photon::pt::pdf_hg(0.5f, 0.f);
      assert(near(p, 1.f / (4.f * photon::pt::PI)));
    }

    {
      int scattered = 0;
      for (int i = 0; i < 256; ++i) {
        Rng rng(1000u + (u32)i);
        Ray r{Vec3{-2.f, 0.f, 0.f}, Vec3{1.f, 0.f, 0.f}};
        const VolumeEvent ev = photon::pt::delta_tracking(vol, r, 0.f, 1e30f, rng);
        if (ev.scattered) {
          scattered++;
          assert(inside(ev.position, vol.bounds_lo, vol.bounds_hi));
        }
      }
      assert(scattered > 160);
    }
  }
  Kokkos::finalize();
  return 0;
}
