#include <anari/anari.h>

#include <cstdio>

static void statusCallback(const void *, ANARIDevice, ANARIObject, ANARIDataType, ANARIStatusSeverity sev,
    ANARIStatusCode, const char *msg)
{
  std::fprintf(stderr, "ANARI(%d): %s\n", int(sev), msg ? msg : "<null>");
}

int main()
{
  auto lib = anariLoadLibrary("opencode_pathtracer", statusCallback, nullptr);
  if (!lib) {
    std::fprintf(stderr, "failed to load library\n");
    return 1;
  }

  std::printf("loaded\n");
  anariUnloadLibrary(lib);
  return 0;
}
