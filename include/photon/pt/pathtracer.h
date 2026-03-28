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

  RenderResult render();

  using PixelView = Kokkos::View<Vec3 **, Kokkos::LayoutRight>;

private:
  void ensure_views(u32 pixel_count, u32 w, u32 h);

  Scene m_scene;
  std::unique_ptr<RayBackend> m_backend_owned;
  RayBackend *m_backend{nullptr};

  // Cached GPU views — allocated once, reused across frames
  u32 m_cached_pixel_count{0};
  u32 m_cached_w{0}, m_cached_h{0};

  // Output views (2D)
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> m_out_color;
  Kokkos::View<f32 **, Kokkos::LayoutRight> m_out_depth;
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> m_out_normal;
  Kokkos::View<Vec3 **, Kokkos::LayoutRight> m_out_albedo;

  // Per-pixel accumulators (1D)
  Kokkos::View<Vec3 *> m_accum;
  Kokkos::View<Vec3 *> m_accum_sq;
  Kokkos::View<u32 *> m_pixel_samples;

  // AOV views
  Kokkos::View<Vec3 *> m_aov_albedo;
  Kokkos::View<Vec3 *> m_aov_normal;
  Kokkos::View<f32 *> m_aov_depth;
  Kokkos::View<u32 *> m_aov_written;

  // Ray batch views
  Kokkos::View<Vec3 *> m_ray_origins;
  Kokkos::View<Vec3 *> m_ray_dirs;
  Kokkos::View<f32 *> m_ray_tmin;
  Kokkos::View<f32 *> m_ray_tmax;

  // Hit batch
  Kokkos::View<HitResult *> m_hits;

  // Path state
  Kokkos::View<Vec3 *> m_throughput;
  Kokkos::View<u32 *> m_active;

  // Shadow ray views
  Kokkos::View<Vec3 *> m_shadow_origins;
  Kokkos::View<Vec3 *> m_shadow_dirs;
  Kokkos::View<f32 *> m_shadow_tmin;
  Kokkos::View<f32 *> m_shadow_tmax;

  // NEE state
  Kokkos::View<Vec3 *> m_nee_contrib;
  Kokkos::View<u32 *> m_nee_active;
  Kokkos::View<u32 *> m_occluded;
  Kokkos::View<Vec3 *> m_sample_start;
};

} // namespace photon::pt
