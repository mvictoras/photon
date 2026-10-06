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

RgbImage rgb_image_from_rgba(const float *rgba, u32 width, u32 height)
{
  RgbImage image{width, height, std::vector<float>(size_t(width) * height * 3)};
  for (size_t i = 0; i < size_t(width) * height; ++i) {
    image.pixels[i * 3 + 0] = rgba[i * 4 + 0];
    image.pixels[i * 3 + 1] = rgba[i * 4 + 1];
    image.pixels[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return image;
}

void write_ppm(const std::string &path, const RgbImageView &image, f32 exposure)
{
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "[photon] write_ppm: cannot open %s for writing\n", path.c_str());
    return;
  }

  out << "P6\n" << image.width << " " << image.height << "\n255\n";

  std::vector<unsigned char> row(size_t(image.width) * 3);
  for (u32 y = 0; y < image.height; ++y) {
    for (u32 x = 0; x < image.width; ++x) {
      const float *px = image.pixels + (size_t(y) * image.width + size_t(x)) * 3;
      const Vec3 c = gamma_correct(aces_tonemap(Vec3{px[0], px[1], px[2]} * exposure));
      row[3 * x + 0] = to_u8(c.x);
      row[3 * x + 1] = to_u8(c.y);
      row[3 * x + 2] = to_u8(c.z);
    }
    out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(row.size()));
  }
}

void write_ppm(const std::string &path, const RenderResult &result, f32 exposure)
{
  auto color_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
  write_ppm(path, RgbImageView{reinterpret_cast<const float *>(color_host.data()),
                               u32(result.color.extent(1)), u32(result.color.extent(0))}, exposure);
}

} // namespace photon::pt::io
