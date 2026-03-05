#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/material.h"

namespace photon::pt {

struct Texture
{
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> pixels;
  u32 width{0};
  u32 height{0};

  KOKKOS_FUNCTION Vec3 sample(f32 u, f32 v) const
  {
    u = u - Kokkos::floor(u);
    v = v - Kokkos::floor(v);

    f32 fx = u * f32(width) - 0.5f;
    f32 fy = v * f32(height) - 0.5f;

    i32 ix = i32(Kokkos::floor(fx));
    i32 iy = i32(Kokkos::floor(fy));
    f32 tx = fx - Kokkos::floor(fx);
    f32 ty = fy - Kokkos::floor(fy);

    auto wrap = [](i32 i, u32 sz) -> u32 {
      i32 s = i32(sz);
      return u32(((i % s) + s) % s);
    };

    u32 x0 = wrap(ix, width);
    u32 x1 = wrap(ix + 1, width);
    u32 y0 = wrap(iy, height);
    u32 y1 = wrap(iy + 1, height);

    Vec3 c00 = pixels(y0, x0);
    Vec3 c10 = pixels(y0, x1);
    Vec3 c01 = pixels(y1, x0);
    Vec3 c11 = pixels(y1, x1);

    Vec3 top = c00 * (1.f - tx) + c10 * tx;
    Vec3 bot = c01 * (1.f - tx) + c11 * tx;
    return top * (1.f - ty) + bot * ty;
  }

  KOKKOS_FUNCTION Vec3 sample_nearest(f32 u, f32 v) const
  {
    u = u - Kokkos::floor(u);
    v = v - Kokkos::floor(v);
    u32 x = u32(u * f32(width)) % width;
    u32 y = u32(v * f32(height)) % height;
    return pixels(y, x);
  }
};

struct TextureAtlas
{
  Kokkos::View<Texture *> textures;
  u32 count{0};

  KOKKOS_FUNCTION Vec3 sample(i32 tex_id, f32 u, f32 v) const
  {
    if (tex_id < 0 || u32(tex_id) >= count)
      return {1.f, 1.f, 1.f};
    return textures(u32(tex_id)).sample(u, v);
  }
};

}
