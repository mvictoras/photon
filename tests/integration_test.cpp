#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/denoiser.h"
#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>

namespace {

using photon::pt::f32;
using photon::pt::u32;
using photon::pt::Vec3;

inline bool finite3(const Vec3 &v)
{
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline f32 len3(const Vec3 &v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    auto scene = SceneBuilder::make_cornell_box();
    auto backend = create_best_backend();
    assert(backend != nullptr);
    backend->build_accel(scene);

    PathTracer pt;
    pt.params.width = 32;
    pt.params.height = 32;
    pt.params.samples_per_pixel = 2;
    pt.params.max_depth = 4;
    pt.camera = Camera::make_perspective({278, 273, -800}, {278, 273, 0}, {0, 1, 0}, 40.f, 1.f);
    pt.set_scene(scene);
    pt.set_backend(std::move(backend));

    auto result = pt.render();

    auto color_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
    auto depth_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.depth);
    auto normal_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.normal);
    auto albedo_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.albedo);

    int nonblack = 0;
    int hits = 0;
    for (u32 y = 0; y < pt.params.height; ++y) {
      for (u32 x = 0; x < pt.params.width; ++x) {
        Vec3 c = color_h(y, x);
        assert(finite3(c));
        if (c.x > 0.0001f || c.y > 0.0001f || c.z > 0.0001f)
          nonblack++;

        f32 d = depth_h(y, x);
        assert(std::isfinite(d));
        if (d > 0.f) {
          hits++;
          Vec3 n = normal_h(y, x);
          assert(finite3(n));
          f32 l = len3(n);
          assert(std::fabs(l - 1.f) < 1e-2f);

          Vec3 a = albedo_h(y, x);
          assert(finite3(a));
          assert(a.x >= 0.f && a.y >= 0.f && a.z >= 0.f);
        } else {
          Vec3 n = normal_h(y, x);
          assert((n.x == 0.f && n.y == 0.f && n.z == 0.f) || finite3(n));
        }
      }
    }

    assert(nonblack > int(pt.params.width * pt.params.height / 8));
    assert(hits > 0);

    bool avail = Denoiser::is_available();
#ifdef PHOTON_HAS_OIDN
    assert(avail);
#else
    assert(!avail);
#endif
  }
  Kokkos::finalize();
  return 0;
}
