#pragma once

#include <anari/backend/DeviceImpl.h>
#include <helium/BaseDevice.h>

#include "photon/anari/runtime/state.h"

namespace photon::anari_device {

struct Device : public helium::BaseDevice
{
  Device(ANARILibrary library);

  ANARIDevice this_device() const;

  void commitParameters(ANARIObject) override;


  int getProperty(ANARIObject, const char *name, ANARIDataType type, void *mem, uint64_t size,
      ANARIWaitMask) override;

  ANARIArray1D newArray1D(const void *, ANARIMemoryDeleter, const void *, ANARIDataType, uint64_t) override;
  ANARIArray2D newArray2D(const void *, ANARIMemoryDeleter, const void *, ANARIDataType, uint64_t, uint64_t) override;
  ANARIArray3D newArray3D(
      const void *, ANARIMemoryDeleter, const void *, ANARIDataType, uint64_t, uint64_t, uint64_t) override;

  ANARIGeometry newGeometry(const char *) override;
  ANARIMaterial newMaterial(const char *) override;
  ANARISampler newSampler(const char *) override;
  ANARISpatialField newSpatialField(const char *) override;
  ANARIVolume newVolume(const char *) override;
  ANARILight newLight(const char *) override;
  ANARISurface newSurface() override;
  ANARIGroup newGroup() override;
  ANARIInstance newInstance(const char *) override;
  ANARIWorld newWorld() override;
  ANARICamera newCamera(const char *) override;
  ANARIRenderer newRenderer(const char *) override;

  const char **getObjectSubtypes(ANARIDataType) override;
  const void *getObjectInfo(ANARIDataType, const char *, const char *, ANARIDataType) override;
  const void *getParameterInfo(
      ANARIDataType, const char *, const char *, ANARIDataType, const char *, ANARIDataType) override;

  ANARIFrame newFrame() override;

 private:
  State m_state;
};

} // namespace photon::anari_device
