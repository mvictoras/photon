#include "photon/anari/SceneFromAnari.h"

#include <cstring>

#include "photon/anari/PhotonDevice.h"

#include <optional>

namespace photon::anari_device {

static bool get_handle_param(const PhotonDevice::Object &o, const char *name, uintptr_t &out)
{
  auto it = o.params.find(name);
  if (it == o.params.end() || it->second.size() != sizeof(uintptr_t))
    return false;
  std::memcpy(&out, it->second.data(), sizeof(uintptr_t));
  return true;
}

std::optional<AnariSceneData> build_scene_from_anari(ANARIWorld world, PhotonDevice &dev)
{
  auto *wo = dev.getObject((uintptr_t)world);
  if (!wo)
    return std::nullopt;

  uintptr_t surfaces_handle = 0;
  if (!get_handle_param(*wo, "surface", surfaces_handle))
    return std::nullopt;

  auto *arr = dev.getObject(surfaces_handle);
  if (!arr || arr->object_type != ANARI_ARRAY1D)
    return std::nullopt;

  return std::nullopt;
}

}
