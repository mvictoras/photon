// Verifies the KHR extension list advertised through anariGetObjectInfo for
// the device. The list must be non-null and null-terminated, must contain the
// five extensions the ANARI CTS requires as a minimum, and must not contain
// any string outside the set of features the device actually implements.

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#include <anari/anari.h>

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void expect(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

bool contains(const char *const *list, const char *name)
{
  for (const char *const *it = list; *it; ++it) {
    if (std::strcmp(*it, name) == 0)
      return true;
  }
  return false;
}

// Every feature the device implements, hand-maintained so the "no phantom
// declaration" check below is independent of the device's own list. When the
// device legitimately adds an extension (src/photon/anari/PhotonDevice.cpp
// getObjectInfo), add it here too — a name advertised by the device but absent
// from this list is treated as a failure.
const char *const implemented[] = {
    "ANARI_KHR_GEOMETRY_TRIANGLE",
    "ANARI_KHR_GEOMETRY_SPHERE",
    "ANARI_KHR_GEOMETRY_CYLINDER",
    "ANARI_KHR_CAMERA_PERSPECTIVE",
    "ANARI_KHR_MATERIAL_MATTE",
    "ANARI_KHR_MATERIAL_PHYSICALLY_BASED",
    "ANARI_KHR_LIGHT_DIRECTIONAL",
    "ANARI_KHR_LIGHT_POINT",
    "ANARI_KHR_LIGHT_QUAD",
    "ANARI_KHR_SAMPLER_IMAGE2D",
    "ANARI_KHR_RENDERER_BACKGROUND_COLOR",
    "ANARI_KHR_RENDERER_AMBIENT_LIGHT",
    "ANARI_KHR_FRAME_CHANNEL_DEPTH",
    "ANARI_KHR_FRAME_CHANNEL_NORMAL",
    "ANARI_KHR_FRAME_CHANNEL_ALBEDO",
    "ANARI_KHR_INSTANCE_TRANSFORM",
    nullptr,
};

bool is_implemented(const char *name)
{
  return contains(implemented, name);
}

} // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    photon::anari_device::PhotonDevice dev(nullptr);

    const auto *extensions = static_cast<const char *const *>(anariGetObjectInfo(
        (ANARIDevice)&dev, ANARI_DEVICE, nullptr, "extension", ANARI_STRING_LIST));

    // Non-null and null-terminated.
    expect(extensions != nullptr, "extension list is null");
    if (extensions) {
      bool terminated = false;
      // Sanity cap well above any plausible list length, not an ANARI limit.
      constexpr size_t kMaxListEntries = 1024;
      for (size_t i = 0; i < kMaxListEntries; ++i) {
        if (extensions[i] == nullptr) {
          terminated = true;
          break;
        }
      }
      expect(terminated, "extension list is not null-terminated");
    }

    // The five extensions the ANARI CTS requires as a minimum.
    expect(contains(extensions, "ANARI_KHR_GEOMETRY_TRIANGLE"),
        "missing ANARI_KHR_GEOMETRY_TRIANGLE");
    expect(contains(extensions, "ANARI_KHR_MATERIAL_MATTE"),
        "missing ANARI_KHR_MATERIAL_MATTE");
    expect(contains(extensions, "ANARI_KHR_CAMERA_PERSPECTIVE"),
        "missing ANARI_KHR_CAMERA_PERSPECTIVE");
    expect(contains(extensions, "ANARI_KHR_RENDERER_BACKGROUND_COLOR"),
        "missing ANARI_KHR_RENDERER_BACKGROUND_COLOR");
    expect(contains(extensions, "ANARI_KHR_RENDERER_AMBIENT_LIGHT"),
        "missing ANARI_KHR_RENDERER_AMBIENT_LIGHT");

    // No phantom declarations: every advertised extension must be a feature
    // the device actually implements.
    if (extensions) {
      for (const char *const *it = extensions; *it; ++it) {
        if (!is_implemented(*it)) {
          std::fprintf(stderr, "FAIL: phantom extension '%s'\n", *it);
          ++failures;
        }
      }
    }
  }
  Kokkos::finalize();

  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  return 0;
}
