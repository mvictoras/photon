#include "photon/pt/denoiser.h"

#include <cstdio>

#ifdef PHOTON_HAS_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace photon::pt {

bool Denoiser::is_available() {
#ifdef PHOTON_HAS_OIDN
  return true;
#else
  return false;
#endif
}

void Denoiser::denoise(
    Kokkos::View<Vec3**, Kokkos::LayoutRight>& color,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& albedo,
    const Kokkos::View<Vec3**, Kokkos::LayoutRight>& normal,
    u32 width, u32 height) {
#ifdef PHOTON_HAS_OIDN
  auto color_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, color);
  auto albedo_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, albedo);
  auto normal_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, normal);

  oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
  device.commit();

  const char* err = nullptr;
  if (device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN device error: %s\n", err ? err : "unknown");
    return;
  }

  const size_t pixel_stride = sizeof(Vec3);
  const size_t row_stride = size_t(width) * pixel_stride;

  oidn::FilterRef filter = device.newFilter("RT");
  filter.setImage("color", color_h.data(), oidn::Format::Float3, width, height, 0, pixel_stride, row_stride);
  filter.setImage("albedo", albedo_h.data(), oidn::Format::Float3, width, height, 0, pixel_stride, row_stride);
  filter.setImage("normal", normal_h.data(), oidn::Format::Float3, width, height, 0, pixel_stride, row_stride);
  filter.setImage("output", color_h.data(), oidn::Format::Float3, width, height, 0, pixel_stride, row_stride);
  filter.set("hdr", true);
  filter.commit();

  if (device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN filter setup error: %s\n", err ? err : "unknown");
    return;
  }

  filter.execute();

  if (device.getError(err) != oidn::Error::None) {
    std::fprintf(stderr, "OIDN execute error: %s\n", err ? err : "unknown");
    return;
  }

  Kokkos::deep_copy(color, color_h);
  std::fprintf(stderr, "  OIDN denoiser completed successfully\n");
#else
  (void)color; (void)albedo; (void)normal; (void)width; (void)height;
#endif
}

}
