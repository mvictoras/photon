#include <anari/backend/LibraryImpl.h>

#include "photon/anari/library.h"

extern "C" {

ANARI_DEFINE_LIBRARY_ENTRYPOINT(photon, handle, scb, scbPtr)
{
  return photon::anari_device::new_library(handle, scb, scbPtr);
}

}
