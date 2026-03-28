#pragma once
#include <Kokkos_Core.hpp>
#include <vector>
#include "photon/pt/math.h"

#ifdef PHOTON_HAS_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace photon::pt {

struct Denoiser {
  // Stateful instance — create once, call denoise_buffer() each frame
  Denoiser();
  ~Denoiser();

  // Denoise flat host buffers in-place (no Kokkos views needed)
  // color: RGB float[w*h*3] (modified in-place)
  // albedo: RGB float[w*h*3] (read-only guide)
  // normal: RGB float[w*h*3] (read-only guide)
  void denoise_buffer(float *color, const float *albedo, const float *normal,
                      u32 width, u32 height);

  // Legacy static API for non-ANARI usage (photon_pbrt_render etc.)
  static void denoise(
    Kokkos::View<Vec3**, Kokkos::LayoutRight>& color,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& albedo,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& normal,
    u32 width, u32 height);

  static bool is_available();

private:
#ifdef PHOTON_HAS_OIDN
  oidn::DeviceRef m_device;
  oidn::FilterRef m_filter;
  std::vector<float> m_output;
  u32 m_width{0};
  u32 m_height{0};
  bool m_ready{false};

  void ensure_setup(u32 width, u32 height);
#endif
};

}
