#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/camera.h"
#include "photon/pt/math.h"
#include "photon/pt/rng.h"
#include "photon/pt/geom/mesh_intersector.h"
#include "photon/pt/scene.h"
#include "photon/pt/sphere.h"

#include <optional>

namespace photon::pt {

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
  std::optional<Scene> scene;

  using PixelView = Kokkos::View<Vec3 **, Kokkos::LayoutRight>;

  PixelView render() const;

 private:
  KOKKOS_FUNCTION Vec3 shade(Rng &rng, Ray ray) const;
};

} // namespace photon::pt
