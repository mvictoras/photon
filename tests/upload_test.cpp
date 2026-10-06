// Unit tests for the shared Scene-upload helpers used by both the PBRT and
// ANARI conversion paths.

#include "photon/pt/scene/upload.h"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    // ---- upload_materials ---------------------------------------------
    {
      std::vector<Material> mats(2);
      mats[0].base_color = {1.f, 0.f, 0.f};
      mats[0].roughness = 0.25f;
      mats[1].base_color = {0.f, 1.f, 0.f};
      mats[1].metallic = 1.f;

      auto view = upload_materials(mats);
      assert(view.extent(0) == 2);
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
      assert(std::fabs(host(0).base_color.x - 1.f) < 1e-6f);
      assert(std::fabs(host(0).roughness - 0.25f) < 1e-6f);
      assert(std::fabs(host(1).base_color.y - 1.f) < 1e-6f);
      assert(std::fabs(host(1).metallic - 1.f) < 1e-6f);
    }

    // ---- upload_lights -------------------------------------------------
    {
      std::vector<Light> lights(1);
      lights[0].type = LightType::Point;
      lights[0].position = {1.f, 2.f, 3.f};
      lights[0].intensity = 4.f;

      auto view = upload_lights(lights);
      assert(view.extent(0) == 1);
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
      assert(host(0).type == LightType::Point);
      assert(std::fabs(host(0).position.z - 3.f) < 1e-6f);
      assert(std::fabs(host(0).intensity - 4.f) < 1e-6f);
    }

    // ---- upload_u32 / upload_f32 ---------------------------------------
    {
      auto ids = upload_u32({7u, 9u, 11u});
      assert(ids.extent(0) == 3);
      auto ids_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ids);
      assert(ids_h(2) == 11u);

      auto areas = upload_f32({0.5f, 1.5f});
      assert(areas.extent(0) == 2);
      auto areas_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, areas);
      assert(std::fabs(areas_h(1) - 1.5f) < 1e-6f);
    }

    // ---- upload_texture_atlas ------------------------------------------
    {
      // Two 2x1 images with distinct texels.
      std::vector<float> img_a = {
          1.f, 0.f, 0.f,   0.f, 1.f, 0.f};
      std::vector<float> img_b = {
          0.f, 0.f, 1.f,   1.f, 1.f, 1.f};

      auto atlas = upload_texture_atlas({{img_a.data(), 2, 1}, {img_b.data(), 2, 1}});

      assert(atlas.count == 2);
      assert(atlas.pixels.extent(0) == 4);
      assert(atlas.infos.extent(0) == 2);

      auto pix_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, atlas.pixels);
      auto info_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, atlas.infos);

      assert(info_h(0).offset == 0 && info_h(0).width == 2 && info_h(0).height == 1);
      assert(info_h(1).offset == 2 && info_h(1).width == 2 && info_h(1).height == 1);
      assert(std::fabs(pix_h(0).x - 1.f) < 1e-6f);
      assert(std::fabs(pix_h(1).y - 1.f) < 1e-6f);
      assert(std::fabs(pix_h(2).z - 1.f) < 1e-6f);
      assert(std::fabs(pix_h(3).x - 1.f) < 1e-6f);
    }

    // ---- upload_texture_atlas: empty input ------------------------------
    {
      auto atlas = upload_texture_atlas({});
      assert(atlas.count == 0);
    }
  }
  Kokkos::finalize();
  return 0;
}
