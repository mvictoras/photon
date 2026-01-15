#pragma once

#include <Kokkos_Core.hpp>

#include "opencode/pt/camera.h"
#include "opencode/pt/math.h"
#include "opencode/pt/rng.h"
#include "opencode/pt/sphere.h"

namespace opencode::pt {

struct RenderParams
{
  u32 width{512};
  u32 height{512};
  u32 samples_per_pixel{8};
  u32 max_depth{5};
};

struct PathTracer
{
  RenderParams params;

  using PixelView = Kokkos::View<Vec3 **, Kokkos::LayoutRight>;

  PixelView render() const;

 private:
  KOKKOS_FUNCTION Vec3 shade(Rng &rng, Ray ray) const;
};

} // namespace opencode::pt
