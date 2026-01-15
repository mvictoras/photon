#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/math_aabb.h"

namespace photon::pt {

inline bool validate_bvh_host(const Bvh &bvh)
{
  auto nodes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bvh.nodes);
  auto prims = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bvh.prim_ids);

  for (size_t i = 0; i < nodes.extent(0); ++i) {
    const auto &n = nodes(i);
    if (n.is_leaf) {
      if (n.count == 0)
        return false;
      if (n.begin + n.count > prims.extent(0))
        return false;
    } else {
      if (n.left >= nodes.extent(0) || n.right >= nodes.extent(0))
        return false;
    }
  }

  return true;
}

} // namespace photon::pt
