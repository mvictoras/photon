#pragma once

#include <anari/anari.h>

namespace opencode::anari_device {

ANARILibrary new_library(void *handle, ANARIStatusCallback scb, const void *scbPtr);

}
