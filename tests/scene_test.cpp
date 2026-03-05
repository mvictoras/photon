#include "photon/pt/scene/builder.h"

#include <Kokkos_Core.hpp>

#include <cassert>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    {
      auto scene = photon::pt::SceneBuilder::make_two_quads();
      assert(scene.mesh.indices.extent(0) > 0);
      assert(scene.mesh.indices.extent(0) % 3 == 0);

      assert(scene.materials.data() != nullptr);
      assert(scene.material_count > 0);
    }

    {
      auto scene = photon::pt::SceneBuilder::make_cornell_box();

      assert(scene.mesh.indices.extent(0) > 0);
      assert(scene.material_count >= 6);

      assert(scene.lights.data() != nullptr);
      assert(scene.light_count >= 1);

      assert(scene.emissive_prim_ids.data() != nullptr);
      assert(scene.emissive_prim_areas.data() != nullptr);
      assert(scene.emissive_count > 0);
      assert(scene.total_emissive_area > 0.f);
    }
  }
  Kokkos::finalize();
  return 0;
}
