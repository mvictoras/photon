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

    Kokkos::View<Vec3 *> results("results", 3);
    Kokkos::parallel_for("tex_test", 1, KOKKOS_LAMBDA(int) {
      results(0) = atlas.sample(0, 0.5f, 0.5f);
      results(1) = atlas.sample(-1, 0.5f, 0.5f);
      results(2) = atlas.sample(99, 0.5f, 0.5f);
    });

    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    assert(std::abs(res_h(0).x - 0.5f) < 0.15f);
    assert(std::abs(res_h(0).y - 0.5f) < 0.15f);
    assert(std::abs(res_h(0).z - 0.5f) < 0.15f);
    assert(std::abs(res_h(1).x - 1.f) < 0.01f);
    assert(std::abs(res_h(2).x - 1.f) < 0.01f);
  }
  Kokkos::finalize();
  return 0;
}
