#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/math.h"

namespace opencode::pt {

struct Material
{
  Vec3 albedo{0.7f, 0.7f, 0.7f};
};

} // namespace opencode::pt
