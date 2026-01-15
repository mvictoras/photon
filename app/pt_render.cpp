#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>

#include "opencode/pt/pathtracer.h"

namespace {

void write_ppm(const std::string &path, const opencode::pt::PathTracer::PixelView &pixels)
{
  const int h = int(pixels.extent(0));
  const int w = int(pixels.extent(1));

  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pixels);

  std::ofstream out(path, std::ios::binary);
  out << "P3\n" << w << " " << h << "\n255\n";

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      auto c = opencode::pt::clamp01(host(y, x));
      const int ir = int(255.999f * c.x);
      const int ig = int(255.999f * c.y);
      const int ib = int(255.999f * c.z);
      out << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    opencode::pt::PathTracer pt;
    pt.params.width = 512;
    pt.params.height = 512;
    pt.params.samples_per_pixel = 32;
    pt.params.max_depth = 8;

    auto pixels = pt.render();
    write_ppm("out.ppm", pixels);
    std::cout << "wrote out.ppm\n";
  }
  Kokkos::finalize();
  return 0;
}
