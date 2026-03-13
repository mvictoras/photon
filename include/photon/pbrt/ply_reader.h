#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include <string>

namespace photon::pbrt {

bool load_ply(const std::string &path, PbrtTriMesh &mesh);

}
