#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include <string>

namespace photon::pbrt {

bool load_image(const std::string &path, PbrtTexture &tex);

}
