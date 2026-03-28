#pragma once

#include <Kokkos_Core.hpp>

#include <memory>
#include <optional>

#include "photon/pt/camera.h"
#include "photon/pt/math.h"
#include "photon/pt/scene.h"
#include "photon/pt/backend/ray_backend.h"

namespace photon::pt {

struct RenderParams {
  u32 width{512};
  u32 height{512};
  u32 samples_per_pixel{8};
  u32 max_depth{5};
  u32 sample_offset{0}; // offset added to RNG seed for accumulation across frames
  Vec3 background_color{0.f, 0.f, 0.f};
  f32 background_alpha{1.f};
  Vec3 ambient_color{1.f, 1.f, 1.f};
  f32 ambient_radiance{0.f};
};

struct RenderResult {
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> color;
  Kokkos::View<f32 **, Kokkos::LayoutRight> depth;
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> normal;
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> albedo;
};

struct PathTracer {
  RenderParams params;
  Camera camera;

  void set_scene(const Scene &scene);
  void set_backend(std::unique_ptr<RayBackend> backend);
  void set_backend_ref(RayBackend &backend);

  RenderResult render() const;

  using PixelView = Kokkos::View<Vec3 **, Kokkos::LayoutRight>;

private:
  Scene m_scene;
  std::unique_ptr<RayBackend> m_backend_owned;
  RayBackend *m_backend{nullptr};
};

} // namespace photon::pt
