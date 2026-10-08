// Verifies the KHR extension list advertised through anariGetObjectInfo for
// the device. The list must be non-null and null-terminated, must contain the
// five extensions the ANARI CTS requires as a minimum, and must not contain
// any string outside the set of features the device actually implements.

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#define ANARI_EXTENSION_UTILITY_IMPL
#include <anari/frontend/anari_extension_utility.h>
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
  if (!list)
    return false;
  for (const char *const *it = list; *it; ++it) {
    if (std::strcmp(*it, name) == 0)
      return true;
  }
  return false;
}

size_t count_entries(const char *const *list)
{
  if (!list)
    return 0;
  size_t count = 0;
  for (const char *const *it = list; *it; ++it)
    ++count;
  return count;
}

size_t count_extension_flags(const ANARIExtensions &extensions)
{
  static_assert(sizeof(ANARIExtensions) % sizeof(int) == 0);
  const auto *flags = reinterpret_cast<const int *>(&extensions);
  size_t count = 0;
  for (size_t i = 0; i < sizeof(ANARIExtensions) / sizeof(int); ++i) {
    if (flags[i] != 0)
      ++count;
  }
  return count;
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

    if (extensions) {
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

      // These capabilities are intentionally omitted until their partial
      // material and sampler implementations are completed.
      expect(!contains(extensions, "ANARI_KHR_MATERIAL_PHYSICALLY_BASED"),
          "advertises partial ANARI_KHR_MATERIAL_PHYSICALLY_BASED");
      expect(!contains(extensions, "ANARI_KHR_SAMPLER_IMAGE2D"),
          "advertises partial ANARI_KHR_SAMPLER_IMAGE2D");

      // Let the ANARI SDK's generated extension table independently reject
      // unknown names. A mismatch also catches duplicate list entries.
      ANARIExtensions known{};
      expect(anariGetObjectExtensionStruct(&known, (ANARIDevice)&dev,
                 ANARI_DEVICE, nullptr) == 0,
          "ANARI extension utility rejected the list");
      // ANARI_KHR_FRAME_CHANNEL_DEPTH is a valid KHR capability but is not
      // represented in this SDK's generated ANARIExtensions struct.
      const size_t sdk_known = count_extension_flags(known);
      const size_t unrepresented = contains(
          extensions, "ANARI_KHR_FRAME_CHANNEL_DEPTH") ? 1 : 0;
      expect(sdk_known + unrepresented == count_entries(extensions),
          "extension list contains an unknown or duplicate name");
    }
  }
  Kokkos::finalize();

  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  return 0;
}
