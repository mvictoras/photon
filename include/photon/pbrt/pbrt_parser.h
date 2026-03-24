#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include <string>

namespace photon::pbrt {

PbrtScene parse_pbrt_file(const std::string &path, bool use_instancing = false);

}
