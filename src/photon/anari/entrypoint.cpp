#include <anari/backend/LibraryImpl.h>

#include "photon/anari/library.h"

#ifdef _WIN32
#define PHOTON_EXPORT __declspec(dllexport)
#else
#define PHOTON_EXPORT __attribute__((visibility("default")))
#endif

extern "C" PHOTON_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(photon, handle, scb, scbPtr)
{
  return photon::anari_device::new_library(handle, scb, scbPtr);
}
