#include "photon/pt/texture.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    TextureAtlas atlas;
    atlas.count = 1;
    atlas.pixels = Kokkos::View<Vec3 *>("pixels", 4);
    atlas.infos = Kokkos::View<TextureInfo *>("infos", 1);

    auto pix_h = Kokkos::create_mirror_view(atlas.pixels);
    pix_h(0) = {1, 0, 0};
    pix_h(1) = {0, 1, 0};
    pix_h(2) = {0, 0, 1};
    pix_h(3) = {1, 1, 1};
    Kokkos::deep_copy(atlas.pixels, pix_h);

    auto info_h = Kokkos::create_mirror_view(atlas.infos);
    info_h(0).offset = 0;
    info_h(0).width = 2;
    info_h(0).height = 2;
    Kokkos::deep_copy(atlas.infos, info_h);

    Vec3 center = atlas.sample(0, 0.5f, 0.5f);
    assert(std::abs(center.x - 0.5f) < 0.15f);
    assert(std::abs(center.y - 0.5f) < 0.15f);
    assert(std::abs(center.z - 0.5f) < 0.15f);

    Vec3 w = atlas.sample(-1, 0.5f, 0.5f);
    assert(std::abs(w.x - 1.f) < 0.01f);
    Vec3 w2 = atlas.sample(99, 0.5f, 0.5f);
    assert(std::abs(w2.x - 1.f) < 0.01f);
  }
  Kokkos::finalize();
  return 0;
}
