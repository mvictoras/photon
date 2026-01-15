#pragma once

#include <anari/anari.h>

namespace photon::anari_device {

ANARILibrary new_library(void *handle, ANARIStatusCallback scb, const void *scbPtr);

} // namespace photon::anari_device
