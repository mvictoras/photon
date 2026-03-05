#include "photon/pt/camera.h"
#include "photon/pt/rng.h"

#include <Kokkos_Core.hpp>

#include <cassert>

using photon::pt::Camera;
using photon::pt::Rng;
using photon::pt::Vec3;
using photon::pt::dot;

static bool approx_eq(photon::pt::f32 a, photon::pt::f32 b, photon::pt::f32 eps = 1e-3f)
{
  return Kokkos::fabs(a - b) <= eps;
}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    const Vec3 look_from{0.f, 0.f, 0.f};
    const Vec3 look_at{0.f, 0.f, -1.f};
    const Vec3 vup{0.f, 1.f, 0.f};

    {
      auto cam = Camera::make_perspective(look_from, look_at, vup, 60.f, 1.f);
      auto r00 = cam.ray(0.f, 0.f);
      auto r11 = cam.ray(1.f, 1.f);
      assert(approx_eq(r00.org.x, look_from.x, 1e-6f));
      assert(approx_eq(r00.org.y, look_from.y, 1e-6f));
      assert(approx_eq(r00.org.z, look_from.z, 1e-6f));
      assert(approx_eq(r11.org.x, look_from.x, 1e-6f));
      assert(approx_eq(r11.org.y, look_from.y, 1e-6f));
      assert(approx_eq(r11.org.z, look_from.z, 1e-6f));
      assert(!approx_eq(dot(r00.dir, r11.dir), 1.f, 1e-3f));
    }

    {
      const photon::pt::f32 focus_dist = 2.f;
      auto cam = Camera::make_perspective(look_from, look_at, vup, 60.f, 1.f, 0.5f, focus_dist);

      Rng rng0(1234u);
      Rng rng1(5678u);
      auto r0 = cam.ray(0.5f, 0.5f, rng0);
      auto r1 = cam.ray(0.5f, 0.5f, rng1);

      assert(!approx_eq(r0.org.x, r1.org.x, 1e-4f) || !approx_eq(r0.org.y, r1.org.y, 1e-4f));

      auto pin = Camera::make_perspective(look_from, look_at, vup, 60.f, 1.f, 0.f, focus_dist);
      auto rp = pin.ray(0.5f, 0.5f);
      const Vec3 focus_pt = rp.org + rp.dir * focus_dist;

      const Vec3 hit0 = r0.org + r0.dir * focus_dist;
      const Vec3 hit1 = r1.org + r1.dir * focus_dist;

      assert(approx_eq(hit0.x, focus_pt.x, 5e-2f));
      assert(approx_eq(hit0.y, focus_pt.y, 5e-2f));
      assert(approx_eq(hit0.z, focus_pt.z, 5e-2f));

      assert(approx_eq(hit1.x, focus_pt.x, 5e-2f));
      assert(approx_eq(hit1.y, focus_pt.y, 5e-2f));
      assert(approx_eq(hit1.z, focus_pt.z, 5e-2f));
    }

    {
      auto cam = Camera::make_orthographic(look_from, look_at, vup, 2.f, 1.f);
      Rng rng(1u);
      auto r0 = cam.ray(0.f, 0.f, rng);
      auto r1 = cam.ray(1.f, 1.f, rng);

      assert(approx_eq(dot(r0.dir, r1.dir), 1.f, 1e-3f));
      assert(!approx_eq(r0.org.x, r1.org.x, 1e-4f) || !approx_eq(r0.org.y, r1.org.y, 1e-4f));
    }
  }
  Kokkos::finalize();
  return 0;
}
