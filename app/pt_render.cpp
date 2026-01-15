#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>

#include "opencode/pt/pathtracer.h"
#include "opencode/pt/scene.h"

namespace {

opencode::pt::Scene make_quad_scene()
{
  using namespace opencode::pt;

  TriangleMesh mesh;
  mesh.positions = Kokkos::View<Vec3 *>("pos", 4);
  mesh.indices = Kokkos::View<u32 *>("idx", 6);
  mesh.albedo_per_prim = Kokkos::View<Vec3 *>("alb", 2);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);

  pos_h(0) = {-1.f, -1.f, -2.5f};
  pos_h(1) = {+1.f, -1.f, -2.5f};
  pos_h(2) = {+1.f, +1.f, -2.5f};
  pos_h(3) = {-1.f, +1.f, -2.5f};

  idx_h(0) = 0;
  idx_h(1) = 1;
  idx_h(2) = 2;
  idx_h(3) = 0;
  idx_h(4) = 2;
  idx_h(5) = 3;

  alb_h(0) = {0.9f, 0.2f, 0.2f};
  alb_h(1) = {0.2f, 0.9f, 0.2f};

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);

  Scene s;
  s.mesh = mesh;
  s.bvh = Bvh::build_cpu(mesh);
  return s;
}


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
    pt.scene = make_quad_scene();

    auto pixels = pt.render();
    write_ppm("out.ppm", pixels);
    std::cout << "wrote out.ppm\n";
  }
  Kokkos::finalize();
  return 0;
}
