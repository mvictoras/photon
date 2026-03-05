#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/scene.h"

namespace photon::pt {

struct SceneBuilder
{
  static Scene make_two_quads();
  static Scene make_cornell_box();
};

} // namespace photon::pt
