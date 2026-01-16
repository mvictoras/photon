#include "photon/anari/device.h"

#include "photon/anari/device_impl.h"

namespace photon::anari_device {

ANARIDevice new_device(ANARILibrary library)
{
  return (ANARIDevice) new Device(library);
}

} // namespace photon::anari_device
