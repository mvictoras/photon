#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

KOKKOS_FUNCTION inline Vec3 linear_to_srgb(Vec3 c)
{
  return {Kokkos::pow(c.x, 1.f / 2.2f), Kokkos::pow(c.y, 1.f / 2.2f), Kokkos::pow(c.z, 1.f / 2.2f)};
}

KOKKOS_FUNCTION inline Vec3 srgb_to_linear(Vec3 c)
{
  return {Kokkos::pow(c.x, 2.2f), Kokkos::pow(c.y, 2.2f), Kokkos::pow(c.z, 2.2f)};
}

KOKKOS_FUNCTION inline f32 luminance(Vec3 c)
{
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

KOKKOS_FUNCTION inline Vec3 aces_tonemap(Vec3 c)
{
  const f32 a = 2.51f;
  const f32 b = 0.03f;
  const f32 d = 0.59f;
  const f32 e = 0.14f;

  auto f = [&](f32 x) {
    const f32 num = x * (a * x + b);
    const f32 den = x * (2.43f * x + d) + e;
    return num / den;
  };

  return clamp01({f(c.x), f(c.y), f(c.z)});
}

}
