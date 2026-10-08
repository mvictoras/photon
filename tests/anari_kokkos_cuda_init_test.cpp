#include <anari/anari.h>
#include <cuda_runtime_api.h>

#include <cstdio>

namespace {

int status_errors = 0;

void status_callback(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity severity, ANARIStatusCode, const char *message)
{
  if (severity >= ANARI_SEVERITY_ERROR) {
    ++status_errors;
    std::fprintf(stderr, "ANARI error: %s\n", message ? message : "<null>");
  }
}

} // namespace

int main()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("photon_anari_kokkos_cuda_init_test: skipped (no CUDA device)\n");
    return 77;
  }

  const int selected_device = device_count - 1;
  if (cudaSetDevice(selected_device) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: could not select CUDA device %d\n", selected_device);
    return 1;
  }

  ANARILibrary library = anariLoadLibrary(
      "photon," PHOTON_ANARI_LIB_DIR, status_callback, nullptr);
  if (!library) {
    std::fprintf(stderr, "FAIL: could not load photon ANARI library\n");
    return 1;
  }

  ANARIDevice device = anariNewDevice(library, "default");
  if (!device) {
    std::fprintf(stderr,
        "FAIL: could not create photon device after cudaSetDevice(%d)\n",
        selected_device);
    anariUnloadLibrary(library);
    return 1;
  }

  int current_device = -1;
  const cudaError_t current_device_result = cudaGetDevice(&current_device);

  anariRelease(device, (ANARIObject)device);
  anariUnloadLibrary(library);

  if (current_device_result != cudaSuccess || current_device != selected_device) {
    std::fprintf(stderr,
        "FAIL: Photon changed the host CUDA device from %d to %d\n",
        selected_device, current_device);
    return 1;
  }

  if (status_errors > 0)
    return 1;

  std::printf(
      "photon_anari_kokkos_cuda_init_test: initialized on CUDA device %d\n",
      selected_device);
  return 0;
}
