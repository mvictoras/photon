#include "photon/anari/SceneFromAnari.h"

#include <cstring>

#include "photon/anari/PhotonDevice.h"

#include <optional>

namespace photon::anari_device {

std::optional<AnariSceneData> build_scene_from_anari(ANARIWorld, const PhotonDevice &)
{
  return std::nullopt;
}

}
