#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"

namespace photon::pt {

struct TriangleMesh
{
  Kokkos::View<Vec3 *> positions;
  Kokkos::View<u32 *> indices;
  Kokkos::View<Vec3 *> albedo_per_prim;
  Kokkos::View<Vec3 *> normals;
  Kokkos::View<Vec2 *> texcoords;
  Kokkos::View<u32 *> material_ids;

  u32 triangle_count() const
  {
    return u32(indices.extent(0) / 3);
  }

  KOKKOS_FUNCTION bool has_normals() const
  {
    return normals.extent(0) > 0;
  }

  KOKKOS_FUNCTION bool has_texcoords() const
  {
    return texcoords.extent(0) > 0;
  }

  KOKKOS_FUNCTION bool has_material_ids() const
  {
    return material_ids.extent(0) > 0;
  }
};

} // namespace photon::pt
