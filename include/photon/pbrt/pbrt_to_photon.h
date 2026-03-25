#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include "photon/pt/camera.h"
#include "photon/pt/scene.h"
#include <map>
#include <string>

namespace photon::pbrt {

struct ConvertedScene {
  photon::pt::Scene scene;
  photon::pt::Camera camera;
  std::map<std::string, unsigned int> mat_name_to_id;
};

ConvertedScene convert_pbrt_scene(const PbrtScene &pbrt, const std::string &base_dir = "");

}
