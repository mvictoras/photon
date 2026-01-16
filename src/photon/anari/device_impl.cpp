#include "photon/anari/device_impl.h"

#include <cstring>

#include "photon/anari/frame.h"

namespace photon::anari_device {

Device::Device(ANARILibrary library) : helium::BaseDevice(library)
{
  m_state.library = library;
}

ANARIDevice Device::this_device() const
{
  return (ANARIDevice)this;
}

void Device::commitParameters(ANARIObject) {}

int Device::getProperty(ANARIObject, const char *name, ANARIDataType type, void *mem, uint64_t size, ANARIWaitMask)
{
  if (!name)
    return 0;

  if (std::strcmp(name, "version") == 0 && type == ANARI_INT32 && size >= sizeof(int)) {
    *(int *)mem = 1;
    return 1;
  }

  return 0;
}

ANARIArray1D Device::newArray1D(const void *, ANARIMemoryDeleter, const void *, ANARIDataType, uint64_t)
{
  return nullptr;
}

ANARIArray2D Device::newArray2D(const void *, ANARIMemoryDeleter, const void *, ANARIDataType, uint64_t, uint64_t)
{
  return nullptr;
}

ANARIArray3D Device::newArray3D(
    const void *, ANARIMemoryDeleter, const void *, ANARIDataType, uint64_t, uint64_t, uint64_t)
{
  return nullptr;
}

ANARIGeometry Device::newGeometry(const char *)
{
  return nullptr;
}

ANARIMaterial Device::newMaterial(const char *)
{
  return nullptr;
}

ANARISampler Device::newSampler(const char *)
{
  return nullptr;
}

ANARISpatialField Device::newSpatialField(const char *)
{
  return nullptr;
}

ANARIVolume Device::newVolume(const char *)
{
  return nullptr;
}

ANARILight Device::newLight(const char *)
{
  return nullptr;
}

ANARISurface Device::newSurface()
{
  return nullptr;
}

ANARIGroup Device::newGroup()
{
  return nullptr;
}

ANARIInstance Device::newInstance(const char *)
{
  return nullptr;
}

ANARIWorld Device::newWorld()
{
  return nullptr;
}

ANARICamera Device::newCamera(const char *)
{
  return nullptr;
}

ANARIRenderer Device::newRenderer(const char *)
{
  return nullptr;
}

const char **Device::getObjectSubtypes(ANARIDataType)
{
  static const char *empty[] = {nullptr};
  return empty;
}

const void *Device::getObjectInfo(ANARIDataType, const char *, const char *, ANARIDataType)
{
  return nullptr;
}

const void *Device::getParameterInfo(ANARIDataType, const char *, const char *, ANARIDataType, const char *,
    ANARIDataType)
{
  return nullptr;
}

ANARIFrame Device::newFrame()
{
  return (ANARIFrame) new Frame(helium::BaseDevice::m_state.get());
}

} // namespace photon::anari_device
