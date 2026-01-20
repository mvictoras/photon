#include <anari/anari.h>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <fstream>
#include <cstdint>

#include "photon/pt/math.h"

static void statusCallback(const void *, ANARIDevice, ANARIObject, ANARIDataType, ANARIStatusSeverity sev,
    ANARIStatusCode, const char *msg)
{
  std::fprintf(stderr, "ANARI(%d): %s\n", int(sev), msg ? msg : "<null>");
}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  auto lib = anariLoadLibrary("photon", statusCallback, nullptr);
  if (!lib) {
    std::fprintf(stderr, "failed to load library\n");
    return 1;
  }

  auto dev = anariNewDevice(lib, "default");
  if (!dev) {
    std::fprintf(stderr, "failed to create device\n");
    anariUnloadLibrary(lib);
    return 2;
  }

  auto frame = anariNewFrame(dev);
  const uint32_t size[2] = {512u, 512u};
  anariSetParameter(dev, frame, "size", ANARI_UINT32_VEC2, size);
  anariCommitParameters(dev, frame);

  auto geom = anariNewGeometry(dev, "triangle");

  const float vertices[] = {
      -0.8f, -0.8f, -2.5f,
      +0.8f, -0.8f, -2.5f,
      0.0f, +0.8f, -2.5f,
  };

  const uint32_t indices[] = {0u, 1u, 2u};

  auto vtx = anariNewArray1D(dev, vertices, nullptr, nullptr, ANARI_FLOAT32_VEC3, 3);
  auto idx = anariNewArray1D(dev, indices, nullptr, nullptr, ANARI_UINT32_VEC3, 1);

  anariSetParameter(dev, geom, "vertex.position", ANARI_ARRAY1D, &vtx);
  anariSetParameter(dev, geom, "primitive.index", ANARI_ARRAY1D, &idx);
  anariCommitParameters(dev, geom);

  auto surf = anariNewSurface(dev);
  anariSetParameter(dev, surf, "geometry", ANARI_GEOMETRY, &geom);
  anariCommitParameters(dev, surf);

  auto surfaces = anariNewArray1D(dev, &surf, nullptr, nullptr, ANARI_SURFACE, 1);

  auto world = anariNewWorld(dev);
  anariSetParameter(dev, world, "surface", ANARI_ARRAY1D, &surfaces);
  anariCommitParameters(dev, world);

  anariSetParameter(dev, frame, "world", ANARI_WORLD, &world);
  anariCommitParameters(dev, frame);

  anariRenderFrame(dev, frame);
  while (!anariFrameReady(dev, frame, ANARI_WAIT))
    ;

  uint32_t w = 0, h = 0;
  ANARIDataType type = ANARI_UNKNOWN;
  auto *ptr = (const float *)anariMapFrame(dev, frame, "color", &w, &h, &type);

  if (!ptr || (type != ANARI_FLOAT32_VEC3 && type != ANARI_FLOAT32_VEC4)) {
    std::fprintf(stderr, "failed to map frame color\n");
    anariRelease(dev, frame);
    anariRelease(dev, dev);
    anariUnloadLibrary(lib);
    return 3;
  }

  std::ofstream out("anari_out.ppm", std::ios::binary);
  out << "P3\n" << w << " " << h << "\n255\n";

  const uint32_t comps = (type == ANARI_FLOAT32_VEC4) ? 4u : 3u;

  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const uint32_t idx = y * w + x;
      photon::pt::Vec3 c{ptr[comps * idx + 0], ptr[comps * idx + 1], ptr[comps * idx + 2]};
      c = photon::pt::clamp01(c);
      const int ir = int(255.999f * c.x);
      const int ig = int(255.999f * c.y);
      const int ib = int(255.999f * c.z);
      out << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }

  anariUnmapFrame(dev, frame, "color");
  anariRelease(dev, frame);
  anariRelease(dev, dev);
  anariUnloadLibrary(lib);

  std::printf("wrote anari_out.ppm\n");
  Kokkos::finalize();
  return 0;
}
