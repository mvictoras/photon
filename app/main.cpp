#include <iostream>

#include "photon/pathtracer/version.h"

int main()
{
  std::cout << "photon v" << photon::version_string() << "\n";
  return 0;
}
