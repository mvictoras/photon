#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/math.h"

namespace opencode::pt {

struct TriangleMesh
{
  Kokkos::View<Vec3 *> positions;
  Kokkos::View<u32 *> indices;

  u32 triangle_count() const
  {
    return u32(indices.extent(0) / 3);
  }
};

} // namespace opencode::pt
