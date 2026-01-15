#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

struct Material
{
  Vec3 albedo{0.7f, 0.7f, 0.7f};
};

} // namespace photon::pt
