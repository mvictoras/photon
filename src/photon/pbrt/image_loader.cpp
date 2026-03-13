#include "photon/pbrt/image_loader.h"

#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_TGA
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_HDR
#include "stb_image.h"

namespace photon::pbrt {

bool load_image(const std::string &path, PbrtTexture &tex)
{
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
