#pragma once
#include <Kokkos_Core.hpp>
#include "photon/pt/math.h"

namespace photon::pt {

KOKKOS_FUNCTION inline Vec3 aces_tonemap(Vec3 c) {
  const f32 a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
  Vec3 num = c * (c * a + Vec3{b, b, b});
  Vec3 den = c * (c * cc + Vec3{d, d, d}) + Vec3{e, e, e};
  return {num.x / den.x, num.y / den.y, num.z / den.z};
}

KOKKOS_FUNCTION inline Vec3 reinhard_tonemap(Vec3 c) {
  return {c.x / (1.f + c.x), c.y / (1.f + c.y), c.z / (1.f + c.z)};
}

KOKKOS_FUNCTION inline Vec3 exposure_adjust(Vec3 c, f32 exposure) {
  f32 mul = Kokkos::pow(2.f, exposure);
  return c * mul;
}

KOKKOS_FUNCTION inline Vec3 gamma_correct(Vec3 c) {
  auto g = [](f32 x) -> f32 {
    return x <= 0.0031308f ? 12.92f * x : 1.055f * Kokkos::pow(x, 1.f / 2.4f) - 0.055f;
  };
  return {g(c.x), g(c.y), g(c.z)};
}

}
