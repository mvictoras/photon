#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/scene.h"

namespace opencode::pt {

struct SceneBuilder
{
  static Scene make_two_quads();
};

} // namespace opencode::pt
