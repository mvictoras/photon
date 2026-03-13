#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "photon/pbrt/pbrt_parser.h"
#include "photon/pbrt/pbrt_to_photon.h"
#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/pathtracer.h"

namespace {

struct Args {
  std::string scene_path;
  std::string output = "pbrt_render.ppm";
  int width = -1;
  int height = -1;
  int spp = -1;
  int max_depth = -1;
};

Args parse_args(int argc, char **argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc)
      a.output = argv[++i];
    else if (arg == "--width" && i + 1 < argc)
      a.width = std::atoi(argv[++i]);
    else if (arg == "--height" && i + 1 < argc)
      a.height = std::atoi(argv[++i]);
    else if (arg == "--spp" && i + 1 < argc)
      a.spp = std::atoi(argv[++i]);
    else if (arg == "--max-depth" && i + 1 < argc)
      a.max_depth = std::atoi(argv[++i]);
    else if (arg[0] != '-')
      a.scene_path = arg;
  }
  return a;
}

void write_ppm(const std::string &file, const photon::pt::RenderResult &result,
    uint32_t w, uint32_t h)
{
  auto color_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);

  std::ofstream out(file, std::ios::binary);
  out << "P6\n" << w << " " << h << "\n255\n";

  std::vector<unsigned char> row(size_t(w) * 3);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      auto c = color_host(y, x);

      auto to_u8 = [](float v) -> unsigned char {
        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        v = std::pow(v, 1.f / 2.2f);
        return static_cast<unsigned char>(v * 255.f + 0.5f);
      };

      row[3 * x + 0] = to_u8(c.x);
      row[3 * x + 1] = to_u8(c.y);
      row[3 * x + 2] = to_u8(c.z);
    }
    out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(row.size()));
  }
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    Args args = parse_args(argc, argv);
    if (args.scene_path.empty()) {
      std::fprintf(stderr, "Usage: photon_pbrt_render <scene.pbrt> [-o output.ppm] [--width W] [--height H] [--spp N]\n");
      Kokkos::finalize();
      return 1;
    }

    std::fprintf(stderr, "Parsing: %s\n", args.scene_path.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto pbrt_scene = photon::pbrt::parse_pbrt_file(args.scene_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    double parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (args.width > 0) pbrt_scene.width = args.width;
    if (args.height > 0) pbrt_scene.height = args.height;
    if (args.spp > 0) pbrt_scene.spp = args.spp;
    if (args.max_depth > 0) pbrt_scene.max_depth = args.max_depth;

    std::fprintf(stderr, "Parsed in %.1f ms: %zu meshes, %zu materials, %dx%d @ %d spp\n",
        parse_ms, pbrt_scene.meshes.size(), pbrt_scene.named_materials.size(),
        pbrt_scene.width, pbrt_scene.height, pbrt_scene.spp);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto converted = photon::pbrt::convert_pbrt_scene(pbrt_scene);
    auto t3 = std::chrono::high_resolution_clock::now();
    double convert_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::fprintf(stderr, "Scene built in %.1f ms: %u triangles, %u materials, %u lights\n",
        convert_ms, converted.scene.mesh.triangle_count(),
        converted.scene.material_count, converted.scene.light_count);

    auto backend = photon::pt::create_best_backend();
    backend->build_accel(converted.scene);

    photon::pt::PathTracer pt;
    pt.params.width = uint32_t(pbrt_scene.width);
    pt.params.height = uint32_t(pbrt_scene.height);
    pt.params.samples_per_pixel = uint32_t(pbrt_scene.spp);
    pt.params.max_depth = uint32_t(pbrt_scene.max_depth);
    pt.camera = converted.camera;
    pt.set_scene(converted.scene);
    pt.set_backend(std::move(backend));

    auto t4 = std::chrono::high_resolution_clock::now();
    auto result = pt.render();
    auto t5 = std::chrono::high_resolution_clock::now();
    double render_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    std::fprintf(stderr, "Rendered in %.1f ms (%.2f fps)\n", render_ms, 1000.0 / render_ms);

    write_ppm(args.output, result, uint32_t(pbrt_scene.width), uint32_t(pbrt_scene.height));
    std::fprintf(stderr, "Output: %s\n", args.output.c_str());
  }
  Kokkos::finalize();
  return 0;
}
