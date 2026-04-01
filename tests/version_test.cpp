#include "photon/pt/version.h"

#include <cassert>

int main()
{
  assert(photon::version_major == 0);
  assert(photon::version_minor == 0);
  assert(photon::version_patch == 1);
  assert(photon::version_string() != nullptr);
  return 0;
}
