#include "photon/pt/tone_mapping.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>

namespace {

using photon::pt::Vec3;

inline bool in_01(float x)
{
  return x >= 0.f && x <= 1.f;
}

inline bool finite(float x)
{
  return std::isfinite(x);
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    Vec3 hdr{10.f, 2.f, 0.5f};

    Vec3 a = photon::pt::aces_tonemap(hdr);
    assert(finite(a.x) && finite(a.y) && finite(a.z));
    assert(in_01(a.x) && in_01(a.y) && in_01(a.z));

    Vec3 r = photon::pt::reinhard_tonemap(hdr);
    assert(finite(r.x) && finite(r.y) && finite(r.z));
    assert(in_01(r.x) && in_01(r.y) && in_01(r.z));
  }
  Kokkos::finalize();
  return 0;
}
