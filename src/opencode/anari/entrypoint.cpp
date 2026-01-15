#include <anari/backend/LibraryImpl.h>

#include "opencode/anari/library.h"

extern "C" {

ANARI_DEFINE_LIBRARY_ENTRYPOINT(opencode_pathtracer, handle, scb, scbPtr)
{
  return opencode::anari_device::new_library(handle, scb, scbPtr);
}

}
