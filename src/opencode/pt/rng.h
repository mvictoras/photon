#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/math.h"

namespace opencode::pt {

struct Rng
{
  u32 state{1u};

  KOKKOS_FUNCTION explicit Rng(u32 seed) : state(seed ? seed : 1u) {}

  KOKKOS_FUNCTION u32 next_u32()
  {
    u32 x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
  }

  KOKKOS_FUNCTION f32 next_f32()
  {
    const u32 x = next_u32();
    return (x >> 8) * (1.0f / 16777216.0f);
  }
};

KOKKOS_FUNCTION inline Vec3 random_in_unit_sphere(Rng &rng)
{
  for (int i = 0; i < 64; ++i) {
    const Vec3 p{2.f * rng.next_f32() - 1.f, 2.f * rng.next_f32() - 1.f, 2.f * rng.next_f32() - 1.f};
    if (dot(p, p) < 1.f)
      return p;
  }
  return {0.f, 0.f, 0.f};
}

KOKKOS_FUNCTION inline Vec3 random_unit_vector(Rng &rng)
{
  return normalize(random_in_unit_sphere(rng));
}

} // namespace opencode::pt
