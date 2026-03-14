#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/material.h"

namespace photon::pt {

struct TextureInfo
{
  u32 offset{0};
  u32 width{0};
  u32 height{0};
  u32 pad{0};
};

struct TextureAtlas
{
  Kokkos::View<Vec3 *> pixels;
  Kokkos::View<TextureInfo *> infos;
  u32 count{0};

  KOKKOS_FUNCTION Vec3 sample(i32 tex_id, f32 u, f32 v) const
  {
    if (tex_id < 0 || u32(tex_id) >= count)
      return {1.f, 1.f, 1.f};

    const TextureInfo ti = infos(u32(tex_id));
    if (ti.width == 0 || ti.height == 0)
      return {1.f, 1.f, 1.f};

    u = u - Kokkos::floor(u);
    v = 1.f - (v - Kokkos::floor(v));

    f32 fx = u * f32(ti.width) - 0.5f;
    f32 fy = v * f32(ti.height) - 0.5f;

    i32 ix = i32(Kokkos::floor(fx));
    i32 iy = i32(Kokkos::floor(fy));
    f32 tx = fx - Kokkos::floor(fx);
    f32 ty = fy - Kokkos::floor(fy);

    auto wrap = [](i32 i, u32 sz) -> u32 {
      i32 s = i32(sz);
      return u32(((i % s) + s) % s);
    };

    u32 x0 = wrap(ix, ti.width);
    u32 x1 = wrap(ix + 1, ti.width);
    u32 y0 = wrap(iy, ti.height);
    u32 y1 = wrap(iy + 1, ti.height);

    const u32 base = ti.offset;
    Vec3 c00 = pixels(base + y0 * ti.width + x0);
    Vec3 c10 = pixels(base + y0 * ti.width + x1);
    Vec3 c01 = pixels(base + y1 * ti.width + x0);
    Vec3 c11 = pixels(base + y1 * ti.width + x1);

    Vec3 top = c00 * (1.f - tx) + c10 * tx;
    Vec3 bot = c01 * (1.f - tx) + c11 * tx;
    return top * (1.f - ty) + bot * ty;
  }
};

}
