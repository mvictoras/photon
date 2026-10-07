// Verifies that the photon ANARI library loads through the public C API and
// reports its implemented KHR extensions — the exact surface hosts such as
// ParaView and the ANARI CTS use. Regression test for extension declarations
// (PhotonDevice::getObjectInfo and Library::getDeviceExtensions).

#include <anari/anari.h>

#include <cstdio>
#include <cstring>

namespace {

void load_status_callback(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity severity, ANARIStatusCode, const char *msg)
{
  if (severity <= ANARI_SEVERITY_ERROR && msg)
    std::fprintf(stderr, "ANARI library status: %s\n", msg);
}

bool contains_extension(const char *const *list, const char *name)
{
  if (!list)
    return false;
  for (const char *const *it = list; *it; ++it) {
    if (std::strcmp(*it, name) == 0)
      return true;
  }
  return false;
}

} // namespace

int main()
{
  ANARILibrary lib =
      anariLoadLibrary("photon," PHOTON_ANARI_LIB_DIR, load_status_callback, nullptr);
  if (!lib) {
    std::fprintf(stderr, "FAIL: could not load photon ANARI library\n");
    return 1;
  }

  ANARIDevice dev = anariNewDevice(lib, "default");
  if (!dev) {
    std::fprintf(stderr, "FAIL: could not create photon ANARI device\n");
    anariUnloadLibrary(lib);
    return 1;
  }

  // --- Device-level extension query (what ParaView's device info panel reads)
  auto exts = static_cast<const char *const *>(anariGetObjectInfo(
      dev, ANARI_DEVICE, nullptr, "extension", ANARI_STRING_LIST));
  if (!exts) {
    std::fprintf(stderr, "FAIL: device extension list is null\n");
    anariRelease(dev, (ANARIObject)dev);
    anariUnloadLibrary(lib);
    return 1;
  }

  int count = 0;
  for (const char *const *it = exts; *it; ++it)
    ++count;
  if (count == 0) {
    std::fprintf(stderr, "FAIL: device extension list is empty\n");
    anariRelease(dev, (ANARIObject)dev);
    anariUnloadLibrary(lib);
    return 1;
  }

  static const char *required[] = {
      // The five extensions the ANARI CTS requires to run at all
      "ANARI_KHR_GEOMETRY_TRIANGLE",
      "ANARI_KHR_MATERIAL_MATTE",
      "ANARI_KHR_CAMERA_PERSPECTIVE",
      "ANARI_KHR_RENDERER_BACKGROUND_COLOR",
      "ANARI_KHR_RENDERER_AMBIENT_LIGHT",
      // Everything else this device implements
      "ANARI_KHR_GEOMETRY_SPHERE",
      "ANARI_KHR_GEOMETRY_CYLINDER",
      "ANARI_KHR_MATERIAL_PHYSICALLY_BASED",
      "ANARI_KHR_LIGHT_DIRECTIONAL",
      "ANARI_KHR_LIGHT_POINT",
      "ANARI_KHR_LIGHT_QUAD",
      "ANARI_KHR_SAMPLER_IMAGE2D",
      "ANARI_KHR_FRAME_CHANNEL_DEPTH",
      "ANARI_KHR_FRAME_CHANNEL_NORMAL",
      "ANARI_KHR_FRAME_CHANNEL_ALBEDO",
      "ANARI_KHR_INSTANCE_TRANSFORM",
  };

  int missing = 0;
  for (const char *name : required) {
    if (!contains_extension(exts, name)) {
      std::fprintf(stderr, "FAIL: device extension list missing %s\n", name);
      ++missing;
    }
  }

  // --- Library-level extension query (what anariGetDeviceExtensionStruct /
  // the CTS reads)
  const char **lib_exts = anariGetDeviceExtensions(lib, "default");
  if (!lib_exts) {
    std::fprintf(stderr, "FAIL: library extension list is null\n");
    ++missing;
  } else {
    for (const char *name : required) {
      if (!contains_extension(lib_exts, name)) {
        std::fprintf(stderr,
            "FAIL: library extension list missing %s\n", name);
        ++missing;
      }
    }
  }

  anariRelease(dev, (ANARIObject)dev);
  anariUnloadLibrary(lib);

  if (missing > 0)
    return 1;

  std::printf("photon_anari_extension_test: %d extensions declared\n", count);
  return 0;
}
