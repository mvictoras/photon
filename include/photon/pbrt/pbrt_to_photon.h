#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include "photon/pt/camera.h"
#include "photon/pt/scene.h"

namespace photon::pbrt {

struct ConvertedScene {
  photon::pt::Scene scene;
  photon::pt::Camera camera;
  photon::pt::InstancedGeometry instanced_geometry;
};

ConvertedScene convert_pbrt_scene(const PbrtScene &pbrt, const std::string &base_dir = "");

}
