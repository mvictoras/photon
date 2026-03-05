#include "photon/pt/denoiser.h"

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

  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  oidn::FilterRef filter = device.newFilter("RT");
  filter.setImage("color", color_h.data(), oidn::Format::Float3, width, height);
  filter.setImage("albedo", albedo_h.data(), oidn::Format::Float3, width, height);
  filter.setImage("normal", normal_h.data(), oidn::Format::Float3, width, height);
  filter.setImage("output", color_h.data(), oidn::Format::Float3, width, height);
  filter.set("hdr", true);
  filter.commit();
  filter.execute();

  Kokkos::deep_copy(color, color_h);
#else
  (void)color; (void)albedo; (void)normal; (void)width; (void)height;
#endif
}

}
