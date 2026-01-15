#include "opencode/pathtracer/version.h"

#include <cassert>

int main()
{
  assert(opencode::pathtracer::version_major == 0);
  assert(opencode::pathtracer::version_minor == 0);
  assert(opencode::pathtracer::version_patch == 1);
  assert(opencode::pathtracer::version_string() != nullptr);
  return 0;
}
