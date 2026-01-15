#include "photon/pt/pathtracer.h"

#include <Kokkos_Core.hpp>

#include <cassert>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    photon::pt::PathTracer pt;
    pt.params.width = 16;
    pt.params.height = 16;
    pt.params.samples_per_pixel = 1;
    pt.params.max_depth = 1;

    auto img = pt.render();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, img);

    auto c = host(0, 0);
    assert(c.x == c.x);
    assert(c.y == c.y);
    assert(c.z == c.z);
  }
  Kokkos::finalize();
  return 0;
}
