#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/ray.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

struct VolumeGrid
{
  Kokkos::View<f32 ***, Kokkos::LayoutRight> density;
  Vec3 bounds_lo{-1.f, -1.f, -1.f};
  Vec3 bounds_hi{1.f, 1.f, 1.f};
  f32 max_density{1.f};
  Vec3 sigma_s{1.f, 1.f, 1.f};
  Vec3 sigma_a{0.1f, 0.1f, 0.1f};
  f32 g{0.f};
  Vec3 emission{0.f, 0.f, 0.f};
  f32 emission_strength{0.f};

  KOKKOS_FUNCTION f32 sample_density(const Vec3 &world_pos) const
  {
    if (density.extent(0) == 0 || density.extent(1) == 0 || density.extent(2) == 0)
      return 0.f;

    const Vec3 bsize = bounds_hi - bounds_lo;
    if (bsize.x <= 0.f || bsize.y <= 0.f || bsize.z <= 0.f)
      return 0.f;

    const Vec3 p = world_pos;
    const f32 u = (p.x - bounds_lo.x) / bsize.x;
    const f32 v = (p.y - bounds_lo.y) / bsize.y;
    const f32 w = (p.z - bounds_lo.z) / bsize.z;

    const f32 uu = Kokkos::fmin(Kokkos::fmax(u, 0.f), 1.f);
    const f32 vv = Kokkos::fmin(Kokkos::fmax(v, 0.f), 1.f);
    const f32 ww = Kokkos::fmin(Kokkos::fmax(w, 0.f), 1.f);

    const f32 nx = f32(density.extent_int(2));
    const f32 ny = f32(density.extent_int(1));
    const f32 nz = f32(density.extent_int(0));

    const f32 gx = uu * (nx - 1.f);
    const f32 gy = vv * (ny - 1.f);
    const f32 gz = ww * (nz - 1.f);

    const int x0 = int(Kokkos::floor(gx));
    const int y0 = int(Kokkos::floor(gy));
    const int z0 = int(Kokkos::floor(gz));

    const int x1 = x0 + 1 < density.extent_int(2) ? x0 + 1 : x0;
    const int y1 = y0 + 1 < density.extent_int(1) ? y0 + 1 : y0;
    const int z1 = z0 + 1 < density.extent_int(0) ? z0 + 1 : z0;

    const f32 tx = gx - f32(x0);
    const f32 ty = gy - f32(y0);
    const f32 tz = gz - f32(z0);

    const f32 d000 = density(z0, y0, x0);
    const f32 d100 = density(z0, y0, x1);
    const f32 d010 = density(z0, y1, x0);
    const f32 d110 = density(z0, y1, x1);
    const f32 d001 = density(z1, y0, x0);
    const f32 d101 = density(z1, y0, x1);
    const f32 d011 = density(z1, y1, x0);
    const f32 d111 = density(z1, y1, x1);

    const f32 d00 = d000 + (d100 - d000) * tx;
    const f32 d10 = d010 + (d110 - d010) * tx;
    const f32 d01 = d001 + (d101 - d001) * tx;
    const f32 d11 = d011 + (d111 - d011) * tx;

    const f32 d0 = d00 + (d10 - d00) * ty;
    const f32 d1 = d01 + (d11 - d01) * ty;

    const f32 d = d0 + (d1 - d0) * tz;

    return Kokkos::fmin(Kokkos::fmax(d, 0.f), max_density);
  }

  KOKKOS_FUNCTION bool intersect_bounds(const Ray &ray, f32 &t_enter, f32 &t_exit) const
  {
    f32 tmin = -1e30f;
    f32 tmax = +1e30f;

    for (int a = 0; a < 3; ++a) {
      const f32 ro = a == 0 ? ray.org.x : (a == 1 ? ray.org.y : ray.org.z);
      const f32 rd = a == 0 ? ray.dir.x : (a == 1 ? ray.dir.y : ray.dir.z);
      const f32 lo = a == 0 ? bounds_lo.x : (a == 1 ? bounds_lo.y : bounds_lo.z);
      const f32 hi = a == 0 ? bounds_hi.x : (a == 1 ? bounds_hi.y : bounds_hi.z);

      if (rd == 0.f) {
        if (ro < lo || ro > hi)
          return false;
        continue;
      }

      const f32 invd = 1.f / rd;
      const f32 t0 = (lo - ro) * invd;
      const f32 t1 = (hi - ro) * invd;

      const f32 tnear = t0 < t1 ? t0 : t1;
      const f32 tfar = t0 < t1 ? t1 : t0;

      tmin = tnear > tmin ? tnear : tmin;
      tmax = tfar < tmax ? tfar : tmax;
      if (tmax < tmin)
        return false;
    }

    t_enter = tmin;
    t_exit = tmax;
    return true;
  }
};

struct VolumeEvent
{
  Vec3 position;
  Vec3 sigma_s;
  Vec3 sigma_a;
  f32 g;
  Vec3 transmittance;
  bool scattered;
};

KOKKOS_FUNCTION inline VolumeEvent delta_tracking(
    const VolumeGrid &vol, const Ray &ray, f32 tmin, f32 tmax, Rng &rng)
{
  f32 t_enter = 0.f;
  f32 t_exit = 0.f;
  if (!vol.intersect_bounds(ray, t_enter, t_exit)) {
    return {ray.at(tmax), Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 0.f}, vol.g, Vec3{1.f, 1.f, 1.f}, false};
  }

  if (t_enter < tmin)
    t_enter = tmin;
  if (t_exit > tmax)
    t_exit = tmax;

  if (t_exit <= t_enter) {
    return {ray.at(t_exit), Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 0.f}, vol.g, Vec3{1.f, 1.f, 1.f}, false};
  }

  const Vec3 sigma_t = vol.sigma_s + vol.sigma_a;
  const f32 sigma_t_max = max_component(sigma_t) * vol.max_density;

  if (sigma_t_max <= 0.f) {
    return {ray.at(t_exit), Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 0.f}, vol.g, Vec3{1.f, 1.f, 1.f}, false};
  }

  f32 t = t_enter;
  for (int iter = 0; iter < 1 << 20; ++iter) {
    const f32 u = rng.next_f32();
    const f32 uu = u > 1e-7f ? u : 1e-7f;
    t += -Kokkos::log(uu) / sigma_t_max;

    if (t > t_exit) {
      return {ray.at(t_exit), Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 0.f}, vol.g, Vec3{1.f, 1.f, 1.f}, false};
    }

    const f32 density = vol.sample_density(ray.at(t));
    const Vec3 sigma_t_real = sigma_t * density;

    const f32 accept = max_component(sigma_t_real) / sigma_t_max;
    if (rng.next_f32() < accept) {
      return {ray.at(t), vol.sigma_s * density, vol.sigma_a * density, vol.g, Vec3{1.f, 1.f, 1.f}, true};
    }
  }

  return {ray.at(t_exit), Vec3{0.f, 0.f, 0.f}, Vec3{0.f, 0.f, 0.f}, vol.g, Vec3{1.f, 1.f, 1.f}, false};
}

KOKKOS_FUNCTION inline Vec3 sample_hg(const Vec3 &wo, f32 g, Rng &rng)
{
  if (Kokkos::fabs(g) < 1e-3f) {
    return sample_uniform_sphere(rng);
  }

  const f32 u = rng.next_f32();
  const f32 s = (1.f - g * g) / (1.f + g - 2.f * g * u);
  f32 cos_theta = (1.f + g * g - s * s) / (2.f * g);
  cos_theta = Kokkos::fmin(1.f, Kokkos::fmax(-1.f, cos_theta));
  const f32 sin_theta = Kokkos::sqrt(Kokkos::fmax(0.f, 1.f - cos_theta * cos_theta));
  const f32 phi = TWO_PI * rng.next_f32();

  Vec3 t{}, b{};
  onb_from_normal(wo, t, b);

  const Vec3 local{sin_theta * Kokkos::cos(phi), sin_theta * Kokkos::sin(phi), cos_theta};
  Vec3 wi = local.x * t + local.y * b + local.z * wo;
  wi = normalize(wi);
  return wi;
}

KOKKOS_FUNCTION inline f32 pdf_hg(f32 cos_theta, f32 g)
{
  if (Kokkos::fabs(g) < 1e-3f)
    return 1.f / (4.f * PI);

  const f32 denom = 1.f + g * g - 2.f * g * cos_theta;
  return INV_PI * 0.25f * (1.f - g * g) / (denom * Kokkos::sqrt(denom));
}

}
