#include "photon/pt/light.h"
#include "photon/pt/environment_map.h"

#include <Kokkos_Core.hpp>

#include <cassert>

namespace {

using photon::pt::EnvironmentMap;
using photon::pt::Light;
using photon::pt::LightSample;
using photon::pt::LightType;
using photon::pt::Rng;
using photon::pt::Vec2;
using photon::pt::Vec3;
using photon::pt::f32;
using photon::pt::u32;

KOKKOS_FUNCTION inline bool near(f32 a, f32 b, f32 eps = 1e-4f)
{
  return Kokkos::fabs(a - b) <= eps;
}

KOKKOS_FUNCTION inline bool near(Vec3 a, Vec3 b, f32 eps = 1e-4f)
{
  return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps);
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {

    {
      Light l;
      l.type = LightType::Point;
      l.position = {0.f, 5.f, 0.f};
      l.color = {1.f, 1.f, 1.f};
      l.intensity = 10.f;

      Rng rng(1u);
      const Vec3 hit{0.f, 0.f, 0.f};
      const LightSample s = photon::pt::sample_light(l, hit, rng);

      assert(near(s.wi, Vec3{0.f, 1.f, 0.f}));
      assert(s.Li.x > 0.f && s.Li.y > 0.f && s.Li.z > 0.f);
      assert(s.pdf > 0.f);
      assert(s.is_delta);
      assert(near(s.dist, 5.f));
    }


    {
      Light l;
      l.type = LightType::Directional;
      l.direction = photon::pt::normalize(Vec3{1.f, -2.f, 3.f});
      l.color = {0.5f, 0.25f, 0.75f};
      l.intensity = 2.f;

      Rng rng(2u);
      const Vec3 hit{1.f, 2.f, 3.f};
      const LightSample s = photon::pt::sample_light(l, hit, rng);

      assert(near(s.wi, -l.direction));
      assert(near(s.Li, l.color * l.intensity));
      assert(s.is_delta);
      assert(s.pdf > 0.f);
      assert(s.dist > 1e20f);
    }


    {
      Light l;
      l.type = LightType::Spot;
      l.position = {0.f, 5.f, 0.f};
      l.direction = {0.f, -1.f, 0.f};
      l.color = {1.f, 1.f, 1.f};
      l.intensity = 10.f;
      l.spot_angle = 0.5f;
      l.spot_falloff = 0.1f;

      Rng rng(3u);
      const Vec3 hit{0.f, 0.f, 0.f};
      const LightSample s = photon::pt::sample_light(l, hit, rng);
      assert(s.Li.x > 0.f);
    }


    {
      Light l;
      l.type = LightType::Spot;
      l.position = {0.f, 5.f, 0.f};
      l.direction = {0.f, -1.f, 0.f};
      l.color = {1.f, 1.f, 1.f};
      l.intensity = 10.f;
      l.spot_angle = 0.2f;
      l.spot_falloff = 0.05f;

      Rng rng(4u);
      const Vec3 hit{10.f, 0.f, 0.f};
      const LightSample s = photon::pt::sample_light(l, hit, rng);
      assert(s.Li.x < 1e-3f && s.Li.y < 1e-3f && s.Li.z < 1e-3f);
    }


    {
      EnvironmentMap env;
      env.width = 4;
      env.height = 2;
      env.pixels = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("env_pixels", env.height, env.width);
      env.marginal_cdf = Kokkos::View<f32 *>("env_marginal", env.height + 1);
      env.conditional_cdf = Kokkos::View<f32 **>("env_conditional", env.height, env.width + 1);

      auto hostPix = Kokkos::create_mirror_view(env.pixels);
      for (u32 y = 0; y < env.height; ++y) {
        for (u32 x = 0; x < env.width; ++x) {
          hostPix(y, x) = {0.f, 0.f, 0.f};
        }
      }

      hostPix(0, 2) = {10.f, 10.f, 10.f};
      Kokkos::deep_copy(env.pixels, hostPix);

      env.build_cdf();


      Kokkos::View<Vec3 *> eval_result("eval_result", 1);
      {
        const Vec2 uv{(2.f + 0.5f) / 4.f, (0.f + 0.5f) / 2.f};
        Kokkos::parallel_for("env_eval", 1, KOKKOS_LAMBDA(int) {
          const Vec3 dir = EnvironmentMap::uv_to_dir(uv);
          eval_result(0) = env.evaluate(dir);
        });
        auto er_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, eval_result);
        assert(er_h(0).x > 1.f && er_h(0).y > 1.f && er_h(0).z > 1.f);
      }

      u32 brightCount = 0;
      Kokkos::parallel_reduce(
          "env_sample",
          Kokkos::RangePolicy<>(0, 200),
          KOKKOS_LAMBDA(const int i, u32 &acc) {
            Rng rng(100u + (u32)i);
            f32 pdf = 0.f;
            const Vec3 dir = env.sample_direction(rng, pdf);
            if (pdf > 0.f) {
              const Vec3 c = env.evaluate(dir);
              if (c.x > 1.f)
                acc++;
            }
          },
          brightCount);

      assert(brightCount > 120);

      {
        Kokkos::View<f32 *> pdf_results("pdf_results", 2);
        Kokkos::parallel_for("env_pdf_test", 1, KOKKOS_LAMBDA(int) {
          Rng rng(999u);
          f32 pdf = 0.f;
          const Vec3 dir = env.sample_direction(rng, pdf);
          pdf_results(0) = pdf;
          pdf_results(1) = env.pdf(dir);
        });
        auto pr_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pdf_results);
        assert(pr_h(0) > 0.f);
        assert(pr_h(1) > 0.f);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
