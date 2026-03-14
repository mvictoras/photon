#include "photon/pbrt/image_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_TGA
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_HDR
#define STBI_ONLY_PNM
#include "stb_image.h"

namespace photon::pbrt {

static bool load_pfm(const std::string &path, PbrtTexture &tex)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open())
    return false;

  std::string magic;
  std::getline(in, magic);
  if (magic != "PF" && magic != "Pf")
    return false;

  int channels = (magic == "PF") ? 3 : 1;

  std::string dim_line;
  std::getline(in, dim_line);
  int w = 0, h = 0;
  std::sscanf(dim_line.c_str(), "%d %d", &w, &h);
  if (w <= 0 || h <= 0)
    return false;

  std::string scale_line;
  std::getline(in, scale_line);
  float scale = 0;
  std::sscanf(scale_line.c_str(), "%f", &scale);
  bool little_endian = (scale < 0);
  float abs_scale = std::fabs(scale);
  if (abs_scale == 0.f)
    abs_scale = 1.f;

  tex.width = w;
  tex.height = h;
  tex.channels = 3;
  tex.data.resize(size_t(w) * size_t(h) * 3);

  size_t row_bytes = size_t(w) * channels * sizeof(float);
  std::vector<float> row_buf(size_t(w) * channels);

  for (int y = h - 1; y >= 0; --y) {
    in.read(reinterpret_cast<char *>(row_buf.data()), std::streamsize(row_bytes));
    for (int x = 0; x < w; ++x) {
      size_t dst = (size_t(y) * w + x) * 3;
      if (channels == 3) {
        tex.data[dst + 0] = row_buf[x * 3 + 0] * abs_scale;
        tex.data[dst + 1] = row_buf[x * 3 + 1] * abs_scale;
        tex.data[dst + 2] = row_buf[x * 3 + 2] * abs_scale;
      } else {
        float v = row_buf[x] * abs_scale;
        tex.data[dst + 0] = v;
        tex.data[dst + 1] = v;
        tex.data[dst + 2] = v;
      }
    }
  }

  std::fprintf(stderr, "  Loaded PFM: %s (%dx%d, %dch)\n", path.c_str(), w, h, channels);
  return true;
}

bool load_image(const std::string &path, PbrtTexture &tex)
{
  if (path.size() >= 4) {
    std::string ext = path.substr(path.size() - 4);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".pfm")
      return load_pfm(path, tex);
  }

  int w = 0, h = 0, c = 0;
  float *fdata = stbi_loadf(path.c_str(), &w, &h, &c, 3);
  if (!fdata) {
    std::fprintf(stderr, "Warning: cannot load texture: %s\n", path.c_str());
    return false;
  }

  tex.width = w;
  tex.height = h;
  tex.channels = 3;
  tex.data.resize(size_t(w) * size_t(h) * 3);
  for (size_t i = 0; i < tex.data.size(); ++i)
    tex.data[i] = fdata[i];

  stbi_image_free(fdata);
  std::fprintf(stderr, "  Loaded texture: %s (%dx%d)\n", path.c_str(), w, h);
  return true;
}

}
