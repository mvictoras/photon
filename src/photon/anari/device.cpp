#include "photon/anari/device.h"

#include "photon/anari/PhotonDevice.h"

namespace photon::anari_device {

ANARIDevice new_device(ANARILibrary library)
{
  return (ANARIDevice) new PhotonDevice(library);
}

} // namespace photon::anari_device
