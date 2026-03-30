// photon_render_test — standalone rendering regression test
//
// Validates that the PathTracer produces correct output using KokkosBackend
// without depending on OptiX/GPU or the ANARI API.
//
// Checks:
//   1. Output dimensions match requested width/height
//   2. Non-black pixel ratio > 50%
//   3. No NaN/Inf values in output
//   4. All finite pixel luminance in [0, 50]
//
// Exit code: 0 = PASS, 1 = FAIL

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"
#include "photon/pt/backend/kokkos_backend.h"

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  int ret = EXIT_FAILURE;
  {
  using namespace photon::pt;

  // --- Build scene ---
  Scene scene = SceneBuilder::make_cornell_box();

  auto backend = std::make_unique<KokkosBackend>();
  backend->build_accel(scene);

  // --- Configure path tracer ---
  constexpr u32 W = 64;
  constexpr u32 H = 64;

  PathTracer pt;
  pt.params.width = W;
  pt.params.height = H;
  pt.params.samples_per_pixel = 4;
  pt.params.max_depth = 2;
  pt.params.sample_offset = 42;
  pt.params.ambient_color = Vec3{1.f, 1.f, 1.f};
  pt.params.ambient_radiance = 0.5f;

  pt.camera = Camera::make_pinhole(
      Vec3{278.f, 273.f, -800.f},  // look_from
      Vec3{278.f, 273.f, 0.f},     // look_at
      Vec3{0.f, 1.f, 0.f},         // vup
      40.f,                         // vfov
      static_cast<f32>(W) / static_cast<f32>(H));

  pt.set_scene(scene);
  pt.set_backend(std::move(backend));

  // --- Render ---
  RenderResult result = pt.render();

  // --- Copy to host ---
  auto h_color =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);

  // --- Validate ---
  bool all_pass = true;

  // Check 1: dimensions
  {
    const bool ok =
        h_color.extent(0) == H && h_color.extent(1) == W;
    std::cout << "[CHECK] Dimensions " << h_color.extent(0) << "x"
              << h_color.extent(1) << " (expected " << H << "x" << W
              << "): " << (ok ? "OK" : "FAIL") << "\n";
    if (!ok) all_pass = false;
  }

  // Check 2: non-black pixels > 50%
  {
    u32 nonblack = 0;
    const u32 total = H * W;
    for (u32 y = 0; y < H; ++y) {
      for (u32 x = 0; x < W; ++x) {
        const Vec3 c = h_color(y, x);
        const f32 lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
        if (lum > 0.001f) ++nonblack;
      }
    }
    const f32 ratio = static_cast<f32>(nonblack) / static_cast<f32>(total);
    const bool ok = ratio > 0.50f;
    std::cout << "[CHECK] Non-black pixels: " << nonblack << "/" << total
              << " (" << (ratio * 100.f) << "%): "
              << (ok ? "OK" : "FAIL") << "\n";
    if (!ok) all_pass = false;
  }

  // Check 3: no NaN/Inf
  {
    u32 bad = 0;
    for (u32 y = 0; y < H; ++y) {
      for (u32 x = 0; x < W; ++x) {
        const Vec3 c = h_color(y, x);
        if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z))
          ++bad;
      }
    }
    const bool ok = (bad == 0);
    std::cout << "[CHECK] NaN/Inf pixels: " << bad << ": "
              << (ok ? "OK" : "FAIL") << "\n";
    if (!ok) all_pass = false;
  }

  // Check 4: luminance range [0, 50]
  {
    u32 out_of_range = 0;
    for (u32 y = 0; y < H; ++y) {
      for (u32 x = 0; x < W; ++x) {
        const Vec3 c = h_color(y, x);
        if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z))
          continue; // skip NaN/Inf (already reported)
        const f32 lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
        if (lum < 0.f || lum > 50.f) ++out_of_range;
      }
    }
    const bool ok = (out_of_range == 0);
    std::cout << "[CHECK] Luminance out of [0,50]: " << out_of_range << ": "
              << (ok ? "OK" : "FAIL") << "\n";
    if (!ok) all_pass = false;
  }

  // --- Final verdict ---
  std::cout << (all_pass ? "PASS" : "FAIL") << "\n";
  ret = all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  Kokkos::finalize();
  return ret;
}
