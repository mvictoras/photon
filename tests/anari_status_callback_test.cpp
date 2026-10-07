// Verifies that ANARI status messages reach the callback registered with
// anariLoadLibrary(). Previously PhotonDevice::report() was a no-op, so hosts
// had no way to see warnings or errors. A render with no world set is a known
// warning path.

#include <anari/anari.h>

#include <cstdio>

namespace {

int warning_count = 0;
int failure_count = 0;

void status_callback(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity severity, ANARIStatusCode, const char *msg)
{
  if (severity >= ANARI_SEVERITY_WARNING) {
    if (!msg) {
      // The status message must be a non-null string (issue #9 criterion).
      ++failure_count;
      std::fprintf(stderr,
          "ANARI status (severity %d) delivered a null message\n",
          int(severity));
      return;
    }
    ++warning_count;
    std::fprintf(stderr, "ANARI status (severity %d): %s\n", int(severity),
        msg);
  }
}

} // namespace

int main()
{
  ANARILibrary lib =
      anariLoadLibrary("photon," PHOTON_ANARI_LIB_DIR, status_callback, nullptr);
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

  ANARIFrame frame = anariNewFrame(dev);
  const uint32_t size[2] = {64, 64};
  anariSetParameter(dev, frame, "size", ANARI_UINT32_VEC2, size);
  anariCommitParameters(dev, frame);

  // Render with no 'world' parameter set — a known warning path.
  anariRenderFrame(dev, frame);

  if (warning_count == 0) {
    std::fprintf(stderr,
        "FAIL: status callback was never invoked for the missing-world "
        "warning\n");
    return 1;
  }

  anariRelease(dev, (ANARIObject)frame);
  anariRelease(dev, (ANARIObject)dev);
  anariUnloadLibrary(lib);

  if (failure_count > 0)
    return 1;

  std::printf("photon_anari_status_callback_test: %d non-null status "
              "message(s) delivered\n",
      warning_count);
  return 0;
}
