#pragma once

#include <anari/anari.h>

namespace photon::anari_device {

struct State
{
  ANARILibrary library{nullptr};
  ANARIStatusCallback status_cb{nullptr};
  const void *status_cb_ptr{nullptr};
};

} // namespace photon::anari_device
