#include "photon/anari/SceneFromAnari.h"

#include <cstring>

#include <Kokkos_Core.hpp>

#include "photon/anari/PhotonDevice.h"

#include <optional>

#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_mesh.h"
#include "photon/pt/math.h"
#include "photon/pt/scene.h"

namespace photon::anari_device {

std::optional<AnariSceneData> build_scene_from_anari(ANARIWorld, const PhotonDevice &)
{
  return std::nullopt;
}

}
