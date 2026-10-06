#pragma once

#include "photon/pt/math.h"
#include "photon/pt/pathtracer.h"

#include <string>
#include <vector>

namespace photon::pt::io {

struct RgbImage {
  u32 width{0};
  u32 height{0};
  std::vector<float> pixels; // row-major, interleaved RGB
};

struct RgbImageView {
  const float *pixels{nullptr};
  u32 width{0};
  u32 height{0};
};

// Convert common MPI/scivis pixel records into the shared RGB image shape.
template <typename Pixel>
RgbImage rgb_image_from_pixels(const std::vector<Pixel> &source,
                               u32 width, u32 height)
{
  RgbImage image{width, height, std::vector<float>(size_t(width) * height * 3)};
  for (size_t i = 0; i < size_t(width) * height; ++i) {
    image.pixels[i * 3 + 0] = source[i].r;
    image.pixels[i * 3 + 1] = source[i].g;
    image.pixels[i * 3 + 2] = source[i].b;
  }
  return image;
}

RgbImage rgb_image_from_rgba(const float *rgba, u32 width, u32 height);

void write_ppm(const std::string &path, const RgbImageView &image,
               f32 exposure = 1.f);

inline void write_ppm(const std::string &path, const RgbImage &image,
                      f32 exposure = 1.f)
{
  write_ppm(path, RgbImageView{image.pixels.data(), image.width, image.height}, exposure);
}

// Convenience overload for a PathTracer RenderResult: mirrors the color AOV
// to the host and writes it.
void write_ppm(const std::string &path, const RenderResult &result,
               f32 exposure = 1.f);

} // namespace photon::pt::io
