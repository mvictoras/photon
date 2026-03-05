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


}
