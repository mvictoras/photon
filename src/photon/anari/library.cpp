#include "photon/anari/library.h"

#include <anari/backend/LibraryImpl.h>

#include <cstring>
#include <string_view>

#include "photon/anari/device.h"
#include "photon/anari/PhotonDevice.h"

namespace photon::anari_device {

namespace {

struct Library : public anari::LibraryImpl
{
  const char **getDeviceExtensions(const char *) override
  {
    // Same list the device reports via anariGetObjectInfo(ANARI_DEVICE,
    // "extension", ANARI_STRING_LIST) — this is what the CTS reads through
    // anariGetDeviceExtensionStruct().
    return photon::anari_device::photon_device_extensions();
  }

  const char **getDeviceSubtypes() override
  {
    static const char *subtypes[] = {"default", nullptr};
    return subtypes;
  }

  Library(void *handle, ANARIStatusCallback scb, const void *scbPtr)
      : anari::LibraryImpl(handle, scb, scbPtr)
  {}

  ANARIDevice newDevice(const char *subtype) override
  {
    if (!subtype || std::string_view(subtype).empty() || std::strcmp(subtype, "default") == 0) {
      return photon::anari_device::new_device(this_library());
    }

    return nullptr;
  }

};

} // namespace

ANARILibrary new_library(void *handle, ANARIStatusCallback scb, const void *scbPtr)
{
  return (ANARILibrary) new Library(handle, scb, scbPtr);
}

} // namespace photon::anari_device
