#pragma once

#include <anari/backend/DeviceImpl.h>
#include <helium/BaseFrame.h>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <string_view>

#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"

namespace photon::anari_device {

struct Frame : public helium::BaseFrame
{
  Frame(helium::BaseGlobalDeviceState *state);

  bool isValid() const override;
  bool getProperty(const std::string_view &, ANARIDataType, void *, uint64_t, uint32_t) override;
  void commitParameters() override;
  void finalize() override;

  void renderFrame() override;
  void *map(std::string_view channel, uint32_t *width, uint32_t *height, ANARIDataType *type) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask) override;
  void discard() override;

 private:
  photon::pt::PathTracer::PixelView m_pixels;
  decltype(Kokkos::create_mirror_view(Kokkos::HostSpace{}, m_pixels)) m_pixels_host;
  bool m_ready{false};
};

} // namespace photon::anari_device
