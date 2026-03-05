#pragma once

#include <anari/anari.h>

#include <optional>
#include <cstdint>

namespace photon::pt { struct Scene; }

namespace photon::anari_device {

struct PhotonDevice;

std::optional<photon::pt::Scene> build_scene_from_anari(ANARIWorld world, const PhotonDevice &dev);

}
