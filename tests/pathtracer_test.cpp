#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"
#include "photon/pt/backend/kokkos_backend.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <memory>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    auto scene = SceneBuilder::make_cornell_box();
    auto backend = std::make_unique<KokkosBackend>();
    backend->build_accel(scene);

    PathTracer pt;
    pt.params.width = 64;
    pt.params.height = 64;
    pt.params.samples_per_pixel = 4;
    pt.params.max_depth = 5;
    pt.camera = Camera::make_perspective({278, 273, -800}, {278, 273, 0}, {0, 1, 0}, 40.f, 1.f);
    pt.set_scene(scene);
    pt.set_backend(std::move(backend));

    auto result = pt.render();

    assert(result.color.extent(0) == 64);
    assert(result.color.extent(1) == 64);
    assert(result.depth.extent(0) == 64);
    assert(result.normal.extent(0) == 64);
    assert(result.albedo.extent(0) == 64);

    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);

    int nonzero = 0;
    for (u32 y = 0; y < 64; ++y)
      for (u32 x = 0; x < 64; ++x) {
        auto c = host(y, x);
        assert(c.x == c.x);
        assert(c.y == c.y);
        assert(c.z == c.z);
        if (c.x > 0.001f || c.y > 0.001f || c.z > 0.001f)
          nonzero++;
      }

    assert(nonzero > 64 * 64 / 4);
  }
  Kokkos::finalize();
  return 0;
}
