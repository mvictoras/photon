#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <cstdint>

#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"
#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/tone_mapping.h"

namespace {

using photon::pt::Vec3;
using photon::pt::f32;
using photon::pt::u32;

inline f32 clamp01(f32 x)
{
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

inline std::uint8_t to_u8(f32 x)
{
  return static_cast<std::uint8_t>(clamp01(x) * 255.999f);
}

inline Vec3 postprocess(Vec3 c)
{
  c = photon::pt::aces_tonemap(c);
  c = photon::pt::clamp01(c);
  c = photon::pt::gamma_correct(c);
  return photon::pt::clamp01(c);
}

void write_ppm(const std::string &path, const photon::pt::PathTracer::PixelView &pixels)
{
  const u32 h = u32(pixels.extent(0));
  const u32 w = u32(pixels.extent(1));

  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pixels);

  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << w << " " << h << "\n255\n";

  for (u32 y = 0; y < h; ++y) {
    for (u32 x = 0; x < w; ++x) {
      Vec3 c = postprocess(host(y, x));
      const std::uint8_t rgb[3] = {to_u8(c.x), to_u8(c.y), to_u8(c.z)};
      out.write(reinterpret_cast<const char *>(rgb), 3);
    }
  }
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    auto scene = photon::pt::SceneBuilder::make_cornell_box();
    auto backend = photon::pt::create_best_backend();
    backend->build_accel(scene);

    photon::pt::PathTracer pt;
    pt.params.width = 512;
    pt.params.height = 512;
    pt.params.samples_per_pixel = 64;
    pt.params.max_depth = 8;
    pt.camera = photon::pt::Camera::make_perspective(
        {278, 273, -800}, {278, 273, 0}, {0, 1, 0},
        40.f, 1.f);
    pt.set_scene(scene);
    pt.set_backend(std::move(backend));

    auto result = pt.render();
    write_ppm("out.ppm", result.color);
    std::cout << "wrote out.ppm\n";
  }
  Kokkos::finalize();
  return 0;
}
