#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/color.h"
#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

struct EnvironmentMap {
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> pixels;
  Kokkos::View<f32 *> marginal_cdf;
  Kokkos::View<f32 **> conditional_cdf;
  u32 width{0};
  u32 height{0};

  // Rotation from world space to texture (env map) space and back.
  f32 world_to_tex[9] = {1,0,0, 0,1,0, 0,0,1};
  f32 tex_to_world[9] = {1,0,0, 0,1,0, 0,0,1};

  KOKKOS_FUNCTION Vec3 to_tex_space(const Vec3 &d) const
  {
    return {world_to_tex[0]*d.x + world_to_tex[1]*d.y + world_to_tex[2]*d.z,
            world_to_tex[3]*d.x + world_to_tex[4]*d.y + world_to_tex[5]*d.z,
            world_to_tex[6]*d.x + world_to_tex[7]*d.y + world_to_tex[8]*d.z};
  }

  KOKKOS_FUNCTION Vec3 to_world_space(const Vec3 &d) const
  {
    return {tex_to_world[0]*d.x + tex_to_world[1]*d.y + tex_to_world[2]*d.z,
            tex_to_world[3]*d.x + tex_to_world[4]*d.y + tex_to_world[5]*d.z,
            tex_to_world[6]*d.x + tex_to_world[7]*d.y + tex_to_world[8]*d.z};
  }

  void build_cdf();

  KOKKOS_FUNCTION Vec3 evaluate(const Vec3 &direction) const
  {
    if (width == 0 || height == 0)
      return {0.f, 0.f, 0.f};

    const Vec2 uv = dir_to_uv(to_tex_space(direction));

    const f32 xf = uv.x * (f32)width;
    const f32 yf = uv.y * (f32)height;

    u32 x = (u32)Kokkos::fmin((f32)(width - 1), Kokkos::fmax(0.f, xf));
    u32 y = (u32)Kokkos::fmin((f32)(height - 1), Kokkos::fmax(0.f, yf));

    return pixels(y, x);
  }

  KOKKOS_FUNCTION Vec3 sample_direction(Rng &rng, f32 &pdf_out) const
  {
    pdf_out = 0.f;
    if (width == 0 || height == 0)
      return {0.f, 1.f, 0.f};

    const f32 u1 = rng.next_f32();
    const f32 u2 = rng.next_f32();

    u32 row = 0;
    {
      u32 lo = 0;
      u32 hi = height;
      while (lo + 1 < hi) {
        const u32 mid = (lo + hi) / 2;
        if (marginal_cdf(mid) <= u2)
          lo = mid;
        else
          hi = mid;
      }
      row = lo;
    }

    u32 col = 0;
    {
      u32 lo = 0;
      u32 hi = width;
      while (lo + 1 < hi) {
        const u32 mid = (lo + hi) / 2;
        if (conditional_cdf(row, mid) <= u1)
          lo = mid;
        else
          hi = mid;
      }
      col = lo;
    }

    const f32 row_cdf0 = marginal_cdf(row);
    const f32 row_cdf1 = marginal_cdf(row + 1);
    const f32 row_pdf = row_cdf1 - row_cdf0;

    const f32 col_cdf0 = conditional_cdf(row, col);
    const f32 col_cdf1 = conditional_cdf(row, col + 1);
    const f32 col_pdf = col_cdf1 - col_cdf0;

    const f32 u = ((f32)col + 0.5f) / (f32)width;
    const f32 v = ((f32)row + 0.5f) / (f32)height;

    const Vec2 uv{u, v};
    const Vec3 dir = to_world_space(uv_to_dir(uv));

    const f32 theta = v * PI;
    const f32 sin_theta = Kokkos::sin(theta);
    const f32 denom = TWO_PI * PI * Kokkos::fmax(sin_theta, 1e-6f);
    pdf_out = (row_pdf * col_pdf) / denom;

    return dir;
  }

  KOKKOS_FUNCTION f32 pdf(const Vec3 &direction) const
  {
    if (width == 0 || height == 0)
      return 0.f;

    const Vec2 uv = dir_to_uv(to_tex_space(direction));
    const f32 xf = uv.x * (f32)width;
    const f32 yf = uv.y * (f32)height;

    u32 x = (u32)Kokkos::fmin((f32)(width - 1), Kokkos::fmax(0.f, xf));
    u32 y = (u32)Kokkos::fmin((f32)(height - 1), Kokkos::fmax(0.f, yf));

    const f32 row_pdf = marginal_cdf(y + 1) - marginal_cdf(y);
    const f32 col_pdf = conditional_cdf(y, x + 1) - conditional_cdf(y, x);

    const f32 theta = ((f32)y + 0.5f) * (PI / (f32)height);
    const f32 sin_theta = Kokkos::sin(theta);
    const f32 denom = TWO_PI * PI * Kokkos::fmax(sin_theta, 1e-6f);
    return (row_pdf * col_pdf) / denom;
  }

  KOKKOS_FUNCTION static Vec2 dir_to_uv(const Vec3 &d)
  {
    const Vec3 nd = normalize(d);
    const f32 phi = Kokkos::atan2(nd.z, nd.x);
    const f32 y = Kokkos::clamp(nd.y, -1.f, 1.f);
    const f32 theta = Kokkos::acos(y);
    const f32 u = (phi + PI) * INV_TWO_PI;
    const f32 v = theta * INV_PI;
    return {u, v};
  }

  KOKKOS_FUNCTION static Vec3 uv_to_dir(const Vec2 &uv)
  {
    const f32 phi = uv.x * TWO_PI - PI;
    const f32 theta = uv.y * PI;
    const f32 sin_theta = Kokkos::sin(theta);
    return {sin_theta * Kokkos::cos(phi), Kokkos::cos(theta), sin_theta * Kokkos::sin(phi)};
  }
};

}
