#pragma once

#include <anari/anari.h>

#include <optional>
#include <cstdint>

#include "photon/pt/scene.h"

namespace photon::anari_device {

struct PhotonDevice;

struct AnariSceneData
{
  photon::pt::Scene scene;
};

std::optional<AnariSceneData> build_scene_from_anari(ANARIWorld world, const PhotonDevice &dev);

}
