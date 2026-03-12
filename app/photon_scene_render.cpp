#include <anari/anari_cpp.hpp>

#include <anari_test_scenes.h>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Args
{
  std::string scene = "demo/cornell_box";
  uint32_t width = 512;
  uint32_t height = 512;
  uint32_t spp = 32;
};

bool parse_u32(const char *s, uint32_t &out)
{
  if (!s)
    return false;
  char *end = nullptr;
  unsigned long v = std::strtoul(s, &end, 10);
  if (!end || *end != '\0')
    return false;
  out = uint32_t(v);
  return true;
}

Args parse_args(int argc, char **argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--scene" && i + 1 < argc) {
      a.scene = argv[++i];
    } else if (arg == "--width" && i + 1 < argc) {
      parse_u32(argv[++i], a.width);
    } else if (arg == "--height" && i + 1 < argc) {
      parse_u32(argv[++i], a.height);
    } else if (arg == "--spp" && i + 1 < argc) {
      parse_u32(argv[++i], a.spp);
    }
  }
  return a;
}

void write_ppm(const std::string &file, const float *rgba, uint32_t w, uint32_t h)
{
  std::ofstream out(file, std::ios::binary);
  out << "P6\n" << w << " " << h << "\n255\n";

  std::vector<unsigned char> row(size_t(w) * 3);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const size_t idx = size_t(y) * size_t(w) + x;
      const float r = rgba[4 * idx + 0];
      const float g = rgba[4 * idx + 1];
      const float b = rgba[4 * idx + 2];

      auto to_u8 = [](float v) {
        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        return (unsigned char)(v * 255.f + 0.5f);
      };

      row[3 * x + 0] = to_u8(r);
      row[3 * x + 1] = to_u8(g);
      row[3 * x + 2] = to_u8(b);
    }
    out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(row.size()));
  }
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    const Args args = parse_args(argc, argv);

    const auto slash = args.scene.find('/');
    const std::string category = slash == std::string::npos ? "demo" : args.scene.substr(0, slash);
    const std::string name = slash == std::string::npos ? args.scene : args.scene.substr(slash + 1);

    anari::Library lib = anari::loadLibrary("photon", nullptr);
    if (!lib)
      return 1;

    anari::Device device = anari::newDevice(lib, "default");
    if (!device)
      return 1;

    anari::commitParameters(device, device);

    auto sceneH = anari::scenes::createScene(device, category.c_str(), name.c_str());
    if (!sceneH)
      return 2;

    anari::scenes::commit(sceneH);
    anari::World world = anari::scenes::getWorld(sceneH);

    auto cameras = anari::scenes::getCameras(sceneH);

    anari::Camera camera = anari::newObject<anari::Camera>(device, "perspective");

    if (!cameras.empty()) {
      anari::setParameter(device, camera, "position", cameras[0].position);
      anari::setParameter(device, camera, "direction", cameras[0].direction);
      anari::setParameter(device, camera, "up", cameras[0].up);
    } else {
      anari::setParameter(device, camera, "position", anari::math::float3(0.f, 0.f, 5.f));
      anari::setParameter(device, camera, "direction", anari::math::float3(0.f, 0.f, -1.f));
      anari::setParameter(device, camera, "up", anari::math::float3(0.f, 1.f, 0.f));
    }

    const float fovy = 0.7f;
    anari::setParameter(device, camera, "fovy", fovy);
    anari::setParameter(device, camera, "aspect", float(args.width) / float(args.height));
    anari::commitParameters(device, camera);

    anari::Renderer renderer = anari::newObject<anari::Renderer>(device, "pathtracer");
    anari::setParameter(device, renderer, "pixelSamples", args.spp);
    anari::setParameter(device, renderer, "maxDepth", uint32_t(8));
    anari::commitParameters(device, renderer);

    anari::Frame frame = anari::newObject<anari::Frame>(device);
    anari::setParameter(device, frame, "size", anari::math::uint2(args.width, args.height));
    anari::setParameter(device, frame, "world", world);
    anari::setParameter(device, frame, "camera", camera);
    anari::setParameter(device, frame, "renderer", renderer);
    anari::commitParameters(device, frame);

    anari::render(device, frame);
    anari::wait(device, frame);

    auto fb = anari::map<anari::math::float4>(device, frame, "color");
    if (!fb.data)
      return 3;

    write_ppm("photon_scene_render.ppm", (const float *)fb.data, fb.width, fb.height);

    anari::unmap(device, frame, "color");

    anari::release(device, frame);
    anari::release(device, renderer);
    anari::release(device, camera);
    anari::scenes::release(sceneH);
    anari::release(device, device);
    anari::unloadLibrary(lib);
  }
  Kokkos::finalize();
  return 0;
}
