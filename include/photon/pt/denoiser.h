#pragma once
#include <Kokkos_Core.hpp>
#include "photon/pt/math.h"

namespace photon::pt {

struct Denoiser {
  static void denoise(
    Kokkos::View<Vec3**, Kokkos::LayoutRight>& color,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& albedo,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& normal,
    u32 width, u32 height);

  static bool is_available();
};

}
