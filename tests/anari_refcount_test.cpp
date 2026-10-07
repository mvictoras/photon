// Exercises array and parameter lifetime handling through the public ANARI C
// API. The API does not expose refcounts, so the test verifies that the full
// host-facing retain/release paths accept valid ownership transitions without
// producing status errors or invalid-handle callbacks.

#include <anari/anari.h>

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;
int status_errors = 0;

void status_callback(const void *, ANARIDevice, ANARIObject,
    ANARIDataType, ANARIStatusSeverity severity, ANARIStatusCode,
    const char *msg)
{
  if (severity >= ANARI_SEVERITY_ERROR) {
    ++status_errors;
    std::fprintf(stderr, "ANARI error: %s\n", msg ? msg : "<null>");
  }
}

void expect(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

} // namespace

int main()
{
  ANARILibrary lib = anariLoadLibrary(
      "photon," PHOTON_ANARI_LIB_DIR, status_callback, nullptr);
  expect(lib != nullptr, "could not load photon ANARI library");
  if (!lib)
    return 1;

  ANARIDevice dev = anariNewDevice(lib, "default");
  expect(dev != nullptr, "could not create photon ANARI device");
  if (!dev) {
    anariUnloadLibrary(lib);
    return 1;
  }

  // A creation-time array owns references to its handle elements.
  ANARISurface surface = anariNewSurface(dev);
  const ANARISurface handles[2] = {surface, nullptr};
  ANARIArray1D array = anariNewArray1D(
      dev, handles, nullptr, nullptr, ANARI_SURFACE, 2);
  expect(surface != nullptr && array != nullptr,
      "could not create surface handle array");
  anariRelease(dev, (ANARIObject)surface);
  anariRelease(dev, (ANARIObject)array);

  // Handle parameters are retained and released through the public API.
  ANARIWorld world = anariNewWorld(dev);
  ANARISurface parameter_surface = anariNewSurface(dev);
  anariSetParameter(dev, world, "surface", ANARI_SURFACE, &parameter_surface);
  anariUnsetParameter(dev, world, "surface");
  anariRelease(dev, (ANARIObject)parameter_surface);
  anariUnsetAllParameters(dev, world);
  anariRelease(dev, (ANARIObject)world);

  // An empty mapped array does not own handles written by the caller.
  ANARISurface mapped_surface = anariNewSurface(dev);
  ANARIArray1D mapped = anariNewArray1D(
      dev, nullptr, nullptr, nullptr, ANARI_SURFACE, 1);
  auto *mapped_memory = static_cast<ANARISurface *>(anariMapArray(dev, mapped));
  expect(mapped_memory != nullptr, "could not map empty handle array");
  if (mapped_memory)
    mapped_memory[0] = mapped_surface;
  anariUnmapArray(dev, mapped);
  anariRelease(dev, (ANARIObject)mapped);
  anariRelease(dev, (ANARIObject)mapped_surface);

  // An 8-byte non-handle parameter must not be mistaken for an object handle.
  ANARIFrame frame = anariNewFrame(dev);
  const uint32_t size[2] = {1, 0};
  anariSetParameter(dev, frame, "size", ANARI_UINT32_VEC2, size);
  anariUnsetAllParameters(dev, frame);
  anariRelease(dev, (ANARIObject)frame);

  expect(status_errors == 0, "lifetime operations reported an ANARI error");
  anariRelease(dev, (ANARIObject)dev);
  anariUnloadLibrary(lib);

  if (failures > 0)
    return 1;

  std::printf("photon_anari_refcount_test: public lifetime checks passed\n");
  return 0;
}
