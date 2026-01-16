#include "photon/anari/frame.h"

#include <helium/BaseGlobalDeviceState.h>

namespace photon::anari_device {

Frame::Frame(helium::BaseGlobalDeviceState *state) : helium::BaseFrame(state) {}

bool Frame::isValid() const
{
  return true;
}

bool Frame::getProperty(const std::string_view &, ANARIDataType, void *, uint64_t, uint32_t)
{
  return false;
}

void Frame::commitParameters()
{
  m_ready = false;
}

void Frame::finalize() {}

void Frame::renderFrame()
{
  const auto sizeParam = getParam<helium::uint2>("size", helium::uint2(512, 512));

  photon::pt::PathTracer pt;
  pt.params.width = sizeParam.x;
  pt.params.height = sizeParam.y;
  pt.params.samples_per_pixel = 8;
  pt.params.max_depth = 4;
  pt.scene = photon::pt::SceneBuilder::make_two_quads();

  m_pixels = pt.render();
  m_pixels_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_pixels);
  m_ready = true;
}

int Frame::frameReady(ANARIWaitMask)
{
  return m_ready ? 1 : 0;
}

void *Frame::map(std::string_view channel, uint32_t *width, uint32_t *height, ANARIDataType *type)
{
  if (!m_ready)
    return nullptr;

  if (channel != "color")
    return nullptr;

  if (width)
    *width = uint32_t(m_pixels_host.extent(1));
  if (height)
    *height = uint32_t(m_pixels_host.extent(0));
  if (type)
    *type = ANARI_FLOAT32_VEC3;

  return (void *)m_pixels_host.data();
}

void Frame::unmap(std::string_view) {}

void Frame::discard()
{
  m_ready = false;
}

} // namespace photon::anari_device
