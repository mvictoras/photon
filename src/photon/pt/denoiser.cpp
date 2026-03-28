#include "photon/pt/denoiser.h"

#include <cstdio>
#include <cstring>

namespace photon::pt {

bool Denoiser::is_available() {
#ifdef PHOTON_HAS_OIDN
  return true;
#else
  return false;
#endif
}

Denoiser::Denoiser() {
#ifdef PHOTON_HAS_OIDN
  // Try CUDA device first for GPU-accelerated denoising, fall back to CPU
  m_device = oidn::newDevice(oidn::DeviceType::Default);
  m_device.commit();

  const char *err = nullptr;
  if (m_device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN: default device failed (%s), trying CPU\n",
                 err ? err : "unknown");
    m_device = oidn::newDevice(oidn::DeviceType::CPU);
    m_device.commit();
    if (m_device.getError(err) != oidn::Error::None) {
      std::fprintf(stderr, "OIDN: CPU device also failed: %s\n",
                   err ? err : "unknown");
      return;
    }
  }
  m_ready = true;
  std::fprintf(stderr, "OIDN: device initialized\n");
#endif
}

Denoiser::~Denoiser() = default;

#ifdef PHOTON_HAS_OIDN
void Denoiser::ensure_setup(u32 width, u32 height) {
  if (m_width == width && m_height == height && m_filter)
    return;

  m_width = width;
  m_height = height;
  m_output.resize(size_t(width) * height * 3);

  m_filter = m_device.newFilter("RT");
  m_filter.set("hdr", true);
  m_filter.set("cleanAux", true);
  // Image pointers are set per-call in denoise_buffer() since the
  // caller's buffer addresses may change between frames.
}
#endif

void Denoiser::denoise_buffer(float *color, const float *albedo,
                               const float *normal, u32 width, u32 height) {
#ifdef PHOTON_HAS_OIDN
  if (!m_ready)
    return;

  ensure_setup(width, height);

  const size_t pixel_stride = 3 * sizeof(float);
  const size_t row_stride = size_t(width) * pixel_stride;

  m_filter.setImage("color", color, oidn::Format::Float3,
                     width, height, 0, pixel_stride, row_stride);
  m_filter.setImage("albedo", const_cast<float *>(albedo),
                     oidn::Format::Float3,
                     width, height, 0, pixel_stride, row_stride);
  m_filter.setImage("normal", const_cast<float *>(normal),
                     oidn::Format::Float3,
                     width, height, 0, pixel_stride, row_stride);
  m_filter.setImage("output", m_output.data(), oidn::Format::Float3,
                     width, height);
  m_filter.commit();

  const char *err = nullptr;
  if (m_device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN filter setup error: %s\n",
                 err ? err : "unknown");
    return;
  }

  m_filter.execute();

  if (m_device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN execute error: %s\n", err ? err : "unknown");
    return;
  }

  // Copy denoised output back to color buffer
  std::memcpy(color, m_output.data(),
              size_t(width) * height * 3 * sizeof(float));
#else
  (void)color; (void)albedo; (void)normal; (void)width; (void)height;
#endif
}

// Legacy static API — creates a temporary Denoiser instance
void Denoiser::denoise(
    Kokkos::View<Vec3**, Kokkos::LayoutRight>& color,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& albedo,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& normal,
    u32 width, u32 height) {
#ifdef PHOTON_HAS_OIDN
  auto color_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, color);
  auto albedo_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, albedo);
  auto normal_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, normal);

  Denoiser dn;
  dn.denoise_buffer(reinterpret_cast<float *>(color_h.data()),
                    reinterpret_cast<const float *>(albedo_h.data()),
                    reinterpret_cast<const float *>(normal_h.data()),
                    width, height);

  Kokkos::deep_copy(color, color_h);
#else
  (void)color; (void)albedo; (void)normal; (void)width; (void)height;
#endif
}

}
