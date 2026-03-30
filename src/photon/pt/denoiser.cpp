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

  const size_t buf_bytes = size_t(width) * height * 3 * sizeof(float);

  // Allocate device-accessible buffers (works for both CPU and GPU devices)
  m_buf_color  = m_device.newBuffer(buf_bytes);
  m_buf_albedo = m_device.newBuffer(buf_bytes);
  m_buf_normal = m_device.newBuffer(buf_bytes);
  m_buf_output = m_device.newBuffer(buf_bytes);

  m_filter = m_device.newFilter("RT");
  m_filter.set("hdr", true);
  m_filter.set("cleanAux", true);

  m_filter.setImage("color",  m_buf_color,  oidn::Format::Float3, width, height);
  m_filter.setImage("albedo", m_buf_albedo, oidn::Format::Float3, width, height);
  m_filter.setImage("normal", m_buf_normal, oidn::Format::Float3, width, height);
  m_filter.setImage("output", m_buf_output, oidn::Format::Float3, width, height);
  m_filter.commit();

  const char *err = nullptr;
  if (m_device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN filter setup error: %s\n",
                 err ? err : "unknown");
  }
}
#endif

void Denoiser::denoise_buffer(float *color, const float *albedo,
                               const float *normal, u32 width, u32 height) {
#ifdef PHOTON_HAS_OIDN
  if (!m_ready)
    return;

  ensure_setup(width, height);

  const size_t buf_bytes = size_t(width) * height * 3 * sizeof(float);

  // Upload host data to device-accessible buffers
  m_buf_color.write(0, buf_bytes, color);
  m_buf_albedo.write(0, buf_bytes, albedo);
  m_buf_normal.write(0, buf_bytes, normal);

  m_filter.execute();

  const char *err = nullptr;
  if (m_device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN execute error: %s\n", err ? err : "unknown");
    return;
  }

  // Read denoised output back to host
  m_buf_output.read(0, buf_bytes, color);
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
