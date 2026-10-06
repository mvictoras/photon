#include "photon/pt/io/ppm.h"

#include "photon/pt/tone_mapping.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace photon::pt::io {

namespace {

unsigned char to_u8(float v)
{
  v = std::clamp(v, 0.f, 1.f);
  return static_cast<unsigned char>(v * 255.f + 0.5f);
}

} // anonymous namespace

void write_ppm(const std::string &path, const float *rgb, u32 width, u32 height,
               f32 exposure)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "[photon] write_ppm: cannot open %s for writing\n", path.c_str());
    return;
  }

  out << "P6\n" << width << " " << height << "\n255\n";

  std::vector<unsigned char> row(size_t(width) * 3);
  for (u32 y = 0; y < height; ++y) {
    for (u32 x = 0; x < width; ++x) {
      const float *px = rgb + (size_t(y) * width + size_t(x)) * 3;
      const Vec3 c = gamma_correct(aces_tonemap(Vec3{px[0], px[1], px[2]} * exposure));
      row[3 * x + 0] = to_u8(c.x);
      row[3 * x + 1] = to_u8(c.y);
      row[3 * x + 2] = to_u8(c.z);
    }
    out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(row.size()));
  }
}

void write_ppm(const std::string &path, const RenderResult &result, u32 width,
               u32 height, f32 exposure)
{
  auto color_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
  write_ppm(path, reinterpret_cast<const float *>(color_host.data()), width, height, exposure);
}

} // namespace photon::pt::io
