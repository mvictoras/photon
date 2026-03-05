#include "photon/pt/disney_bsdf.h"
#include "photon/pt/material.h"
#include "photon/pt/math.h"
#include "photon/pt/rng.h"

#include <Kokkos_Core.hpp>

#include <cassert>

namespace {

using photon::pt::BsdfSample;
using photon::pt::Material;
using photon::pt::Rng;
using photon::pt::Vec3;
using photon::pt::dot;
using photon::pt::f32;
using photon::pt::normalize;

KOKKOS_FUNCTION inline bool near(f32 a, f32 b, f32 eps)
{
  return Kokkos::fabs(a - b) <= eps;
}

KOKKOS_FUNCTION inline bool near_vec3(Vec3 a, Vec3 b, f32 eps)
{
  return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps);
}

KOKKOS_FUNCTION inline bool all_geq(Vec3 v, f32 x)
{
  return v.x >= x && v.y >= x && v.z >= x;
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    const Vec3 n = normalize(Vec3{0.f, 0.f, 1.f});
    const Vec3 shading_n = n;
    const Vec3 wo = normalize(Vec3{0.2f, 0.1f, 0.97f});

    {
      Material mat{};
      Rng rng(1u);
      BsdfSample s = photon::pt::disney_bsdf_sample(mat, wo, n, shading_n, rng);
      assert(dot(s.wi, n) >= -1e-6f);
      assert(s.pdf > 0.f);
      assert(all_geq(s.f, 0.f));
      assert(s.f.x + s.f.y + s.f.z > 0.f);
    }

    {
      Material mat{};
      mat.metallic = 1.f;
      mat.roughness = 0.1f;
      Rng rng(2u);
      BsdfSample s = photon::pt::disney_bsdf_sample(mat, wo, n, shading_n, rng);
      assert(dot(s.wi, n) >= -1e-6f);
      assert(s.pdf > 0.f);
    }

    {
      Material mat{};
      mat.transmission = 1.f;
      mat.ior = 1.5f;
      mat.roughness = 0.f;
      Rng rng(3u);
      BsdfSample s = photon::pt::disney_bsdf_sample(mat, wo, n, shading_n, rng);
      assert(s.pdf > 0.f);
      assert(s.is_specular);
    }

    {
      Material mat{};
      mat.emission = Vec3{1.f, 2.f, 3.f};
      mat.emission_strength = 4.f;
      const Vec3 e = photon::pt::material_emission(mat);
      assert(near_vec3(e, Vec3{4.f, 8.f, 12.f}, 1e-6f));
      assert(photon::pt::material_is_emissive(mat));
    }

    {
      Material mat{};
      mat.base_color = Vec3{1.f, 1.f, 1.f};
      mat.metallic = 0.f;
      mat.transmission = 0.f;
      mat.roughness = 1.f;
      mat.clearcoat = 0.f;
      mat.clearcoat_roughness = 0.03f;

      Rng rng(4u);
      const int N = 10000;
      Vec3 sum{0.f, 0.f, 0.f};

      for (int i = 0; i < N; ++i) {
        BsdfSample s = photon::pt::disney_bsdf_sample(mat, wo, n, shading_n, rng);
        if (s.pdf > 0.f)
          sum += s.f / s.pdf;
      }

      const Vec3 mean = sum / f32(N);
      const f32 expected = 1.f;
      assert(mean.x >= 0.8f * expected && mean.x <= 1.2f * expected);
      assert(mean.y >= 0.8f * expected && mean.y <= 1.2f * expected);
      assert(mean.z >= 0.8f * expected && mean.z <= 1.2f * expected);
    }

    {
      Material mat{};
      Rng rng(5u);
      for (int i = 0; i < 256; ++i) {
        BsdfSample s = photon::pt::disney_bsdf_sample(mat, wo, n, shading_n, rng);
        assert(s.pdf >= 0.f);
        const f32 pdf_eval = photon::pt::disney_bsdf_pdf(mat, wo, s.wi, n, shading_n);
        assert(near(pdf_eval, s.pdf, 5e-4f));
        const Vec3 f_eval = photon::pt::disney_bsdf_eval(mat, wo, s.wi, n, shading_n);
        assert(all_geq(f_eval, 0.f));
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
