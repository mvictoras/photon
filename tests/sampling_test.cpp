#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

#include <Kokkos_Core.hpp>

#include <cassert>

namespace {

using photon::pt::f32;
using photon::pt::Vec2;
using photon::pt::Vec3;

KOKKOS_FUNCTION inline bool near(f32 a, f32 b, f32 eps = 1e-4f)
{
  return Kokkos::fabs(a - b) <= eps;
}

KOKKOS_FUNCTION inline bool near_vec3(Vec3 a, Vec3 b, f32 eps = 1e-4f)
{
  return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps);
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using photon::pt::Rng;
    using photon::pt::dot;
    using photon::pt::length;
    using photon::pt::normalize;


    assert(near(photon::pt::power_heuristic(1, 0.5f, 1, 0.5f), 0.5f));
    assert(near(photon::pt::power_heuristic(1, 1.f, 1, 0.f), 1.f));


    {
      Vec3 n = normalize(Vec3{0.3f, -0.7f, 0.64f});
      Vec3 t{}, b{};
      photon::pt::onb_from_normal(n, t, b);
      assert(near(length(t), 1.f, 2e-4f));
      assert(near(length(b), 1.f, 2e-4f));
      assert(near(dot(n, t), 0.f, 2e-4f));
      assert(near(dot(n, b), 0.f, 2e-4f));
      assert(near(dot(t, b), 0.f, 2e-4f));


      Vec3 c = photon::pt::cross(t, b);
      assert(dot(c, n) > 0.f);
    }

    Rng rng(1337u);


    {
      for (int i = 0; i < 256; ++i) {
        auto bc = photon::pt::sample_uniform_triangle(rng);
        assert(bc.u >= 0.f);
        assert(bc.v >= 0.f);
        assert(bc.u + bc.v <= 1.f + 1e-6f);
      }
    }


    {
      for (int i = 0; i < 256; ++i) {
        Vec2 d = photon::pt::sample_concentric_disk(rng);
        const f32 r2 = d.x * d.x + d.y * d.y;
        assert(r2 <= 1.f + 1e-5f);
      }
    }


    {
      for (int i = 0; i < 256; ++i) {
        Vec3 s = photon::pt::sample_uniform_sphere(rng);
        assert(near(length(s), 1.f, 2e-4f));
      }
    }


    {
      const Vec3 n = normalize(Vec3{0.1f, 0.9f, 0.4f});
      for (int i = 0; i < 256; ++i) {
        auto res = photon::pt::sample_cosine_hemisphere(n, rng);
        const f32 cosTheta = dot(res.direction, n);
        assert(cosTheta >= -1e-6f);
        assert(near(res.pdf, photon::pt::cosine_hemisphere_pdf(cosTheta), 2e-5f));
      }
    }


    {
      const Vec3 n = normalize(Vec3{0.f, 0.f, 1.f});
      f32 sum = 0.f;
      const int N = 1000;
      for (int i = 0; i < N; ++i) {
        auto res = photon::pt::sample_cosine_hemisphere(n, rng);
        sum += dot(res.direction, n);
      }
      const f32 mean = sum / f32(N);
      assert(near(mean, 2.f / 3.f, 0.05f));
    }
  }
  Kokkos::finalize();
  return 0;
}
