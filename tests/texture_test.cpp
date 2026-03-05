#include "photon/pt/texture.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    Texture tex;
    tex.width = 2;
    tex.height = 2;
    tex.pixels = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("tex", 2, 2);
    auto h = Kokkos::create_mirror_view(tex.pixels);
    h(0, 0) = {1, 0, 0};
    h(0, 1) = {0, 1, 0};
    h(1, 0) = {0, 0, 1};
    h(1, 1) = {1, 1, 1};
    Kokkos::deep_copy(tex.pixels, h);

    Vec3 tl = tex.sample_nearest(0.1f, 0.1f);
    assert(tl.x > 0.9f && tl.y < 0.1f);

    Vec3 center = tex.sample(0.5f, 0.5f);
    assert(std::abs(center.x - 0.5f) < 0.15f);
    assert(std::abs(center.y - 0.5f) < 0.15f);
    assert(std::abs(center.z - 0.5f) < 0.15f);

    Vec3 wrapped = tex.sample(1.5f, 0.5f);
    Vec3 normal = tex.sample(0.5f, 0.5f);
    assert(std::abs(wrapped.x - normal.x) < 0.01f);

    TextureAtlas atlas;
    atlas.count = 1;
    atlas.textures = Kokkos::View<Texture *>("atlas", 1);
    auto atlas_h = Kokkos::create_mirror_view(atlas.textures);
    atlas_h(0) = tex;
    Kokkos::deep_copy(atlas.textures, atlas_h);

    Vec3 s = atlas.sample(0, 0.5f, 0.5f);
    assert(std::abs(s.x - 0.5f) < 0.15f);

    Vec3 w = atlas.sample(-1, 0.5f, 0.5f);
    assert(std::abs(w.x - 1.f) < 0.01f);
    Vec3 w2 = atlas.sample(99, 0.5f, 0.5f);
    assert(std::abs(w2.x - 1.f) < 0.01f);
  }
  Kokkos::finalize();
  return 0;
}
