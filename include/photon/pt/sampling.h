#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/rng.h"

namespace photon::pt {

constexpr f32 PI = 3.14159265358979323846f;
constexpr f32 INV_PI = 1.f / PI;
constexpr f32 TWO_PI = 2.f * PI;
constexpr f32 INV_TWO_PI = 1.f / TWO_PI;

KOKKOS_FUNCTION inline f32 power_heuristic(int nf, f32 fPdf, int ng, f32 gPdf)
{
  f32 f = nf * fPdf;
  f32 g = ng * gPdf;
  f32 f2 = f * f;
  f32 g2 = g * g;
  return (f2 + g2) > 0.f ? f2 / (f2 + g2) : 0.f;
}

KOKKOS_FUNCTION inline void onb_from_normal(const Vec3 &n, Vec3 &tangent, Vec3 &bitangent)
{
  if (n.z < -0.9999999f) {
    tangent = {0.f, -1.f, 0.f};
    bitangent = {-1.f, 0.f, 0.f};
    return;
  }

  const f32 a = 1.f / (1.f + n.z);
  const f32 b = -n.x * n.y * a;
  tangent = {1.f - n.x * n.x * a, b, -n.x};
  bitangent = {b, 1.f - n.y * n.y * a, -n.y};

  tangent = normalize(tangent);
  bitangent = normalize(bitangent);
}

struct SampleResult
{
  Vec3 direction;
  f32 pdf{0.f};
};

KOKKOS_FUNCTION inline f32 cosine_hemisphere_pdf(f32 cos_theta)
{
  return Kokkos::fmax(cos_theta, 0.f) * INV_PI;
}

KOKKOS_FUNCTION inline Vec2 sample_concentric_disk(Rng &rng)
{
  const f32 u = 2.f * rng.next_f32() - 1.f;
  const f32 v = 2.f * rng.next_f32() - 1.f;

  if (u == 0.f && v == 0.f)
    return {0.f, 0.f};

  f32 r = 0.f;
  f32 theta = 0.f;

  const f32 au = Kokkos::fabs(u);
  const f32 av = Kokkos::fabs(v);

  if (au > av) {
    r = au;
    theta = (PI / 4.f) * (v / u);
  } else {
    r = av;
    theta = (PI / 2.f) - (PI / 4.f) * (u / v);
  }

  return {r * Kokkos::cos(theta), r * Kokkos::sin(theta)};
}

KOKKOS_FUNCTION inline SampleResult sample_cosine_hemisphere(const Vec3 &n, Rng &rng)
{
  Vec3 t{}, b{};
  onb_from_normal(n, t, b);

  const Vec2 d = sample_concentric_disk(rng);
  const f32 z = Kokkos::sqrt(Kokkos::fmax(0.f, 1.f - d.x * d.x - d.y * d.y));

  const Vec3 local{d.x, d.y, z};
  Vec3 dir = local.x * t + local.y * b + local.z * n;
  dir = normalize(dir);

  const f32 cosTheta = dot(dir, n);
  return {dir, cosine_hemisphere_pdf(cosTheta)};
}

KOKKOS_FUNCTION inline Vec3 sample_uniform_sphere(Rng &rng)
{
  const f32 z = 1.f - 2.f * rng.next_f32();
  const f32 r = Kokkos::sqrt(Kokkos::fmax(0.f, 1.f - z * z));
  const f32 phi = TWO_PI * rng.next_f32();
  return {r * Kokkos::cos(phi), r * Kokkos::sin(phi), z};
}

struct BaryResult
{
  f32 u{0.f};
  f32 v{0.f};
};

KOKKOS_FUNCTION inline BaryResult sample_uniform_triangle(Rng &rng)
{
  const f32 r1 = rng.next_f32();
  const f32 sr1 = Kokkos::sqrt(r1);
  const f32 r2 = rng.next_f32();
  return {1.f - sr1, sr1 * (1.f - r2)};
}

}
