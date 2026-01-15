#pragma once

#include "opencode/pt/math.h"

namespace opencode::pt {

struct Ray
{
  Vec3 org;
  Vec3 dir;

  KOKKOS_FUNCTION Ray() = default;
  KOKKOS_FUNCTION Ray(const Vec3 &o, const Vec3 &d) : org(o), dir(d) {}

  KOKKOS_FUNCTION Vec3 at(f32 t) const
  {
    return org + dir * t;
  }
};

} // namespace opencode::pt
