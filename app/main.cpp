#include <iostream>

#include "opencode/pathtracer/version.h"

int main()
{
  std::cout << "opencode-pathtracer v" << opencode::pathtracer::version_string() << "\n";
  return 0;
}
