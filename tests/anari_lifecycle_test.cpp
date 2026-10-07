// Acceptance tests for issue #9, observed through the device's object table.
// The public C API has no "is alive" query, so refcount-zero checks go through
// PhotonDevice::getObject() directly (same prior art as anari_import_test):
// a freed object's handle no longer resolves.
//
// Failures are counted, not asserted: the Release build defines NDEBUG, which
// would compile plain assert() out and make every check here vacuous.

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#include <anari/anari.h>

#include <cstdint>
#include <cstdio>

namespace {

using photon::anari_device::PhotonDevice;

int failures = 0;

void expect(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

// Releasing an array of surface handles frees the contained surfaces.
void array_of_handles_release_frees_elements(PhotonDevice &dev)
{
  ANARISurface surface = dev.newSurface();
  expect(surface != nullptr, "could not create surface");

  const ANARISurface handles[2] = {surface, nullptr};
  ANARIArray1D array =
      dev.newArray1D(handles, nullptr, nullptr, ANARI_SURFACE, 2);
  expect(array != nullptr, "could not create surface handle array");

  // Drop the application's own reference; the array still holds one.
  dev.release(surface);
  expect(dev.getObject((uintptr_t)surface) != nullptr,
      "surface freed while the array holding it is still alive");

  // Refcount of the array reaches zero: contained handles must be released
  // before the buffer is freed, dropping the surface's refcount to zero too.
  dev.release(array);
  expect(dev.getObject((uintptr_t)surface) == nullptr,
      "surface in released handle-array was not freed");
}

// unsetParameter() releases the handle stored in the parameter slot.
void unset_parameter_releases_handle(PhotonDevice &dev)
{
  ANARIWorld world = dev.newWorld();
  ANARISurface surface = dev.newSurface();

  dev.setParameter(world, "surface", ANARI_SURFACE, &surface);
  dev.unsetParameter(world, "surface");
  dev.release(surface);

  expect(dev.getObject((uintptr_t)surface) == nullptr,
      "unsetParameter did not release the handle parameter");
  dev.release(world);
}

// unsetAllParameters() releases every handle parameter the object holds.
void unset_all_parameters_releases_handles(PhotonDevice &dev)
{
  ANARIWorld world = dev.newWorld();
  ANARISurface surface = dev.newSurface();
  ANARIGeometry geometry = dev.newGeometry("triangle");

  dev.setParameter(world, "surface", ANARI_SURFACE, &surface);
  dev.setParameter(world, "geometry", ANARI_GEOMETRY, &geometry);

  dev.unsetAllParameters(world);
  dev.release(surface);
  dev.release(geometry);

  expect(dev.getObject((uintptr_t)surface) == nullptr,
      "unsetAllParameters did not release 'surface'");
  expect(dev.getObject((uintptr_t)geometry) == nullptr,
      "unsetAllParameters did not release 'geometry'");
  dev.release(world);
}

int32_t query_num_samples(PhotonDevice &dev, ANARIFrame frame)
{
  int32_t num_samples = -1;
  if (!dev.getProperty(frame, "numSamples", ANARI_INT32, &num_samples,
          sizeof(num_samples), ANARI_NO_WAIT)) {
    expect(false, "could not query numSamples");
    return -1;
  }
  return num_samples;
}

// renderFrame() uses the exact pixelSamples value set via anariSetParameter.
void pixel_samples_int32_is_used_exactly(PhotonDevice &dev)
{
  ANARIRenderer renderer = dev.newRenderer("default");
  const int32_t spp = 7;
  dev.setParameter(renderer, "pixelSamples", ANARI_INT32, &spp);
  dev.commitParameters(renderer);

  ANARIFrame frame = dev.newFrame();
  const uint32_t size[2] = {32, 32};
  dev.setParameter(frame, "size", ANARI_UINT32_VEC2, size);
  dev.setParameter(frame, "renderer", ANARI_RENDERER, &renderer);
  dev.commitParameters(frame);

  dev.renderFrame(frame);
  expect(query_num_samples(dev, frame) == spp,
      "renderFrame did not use pixelSamples exactly");

  // Accumulation continues with the same sample count.
  dev.renderFrame(frame);
  expect(query_num_samples(dev, frame) == 2 * spp,
      "second renderFrame did not add pixelSamples again");

  dev.release(frame);
  dev.release(renderer);
}

} // namespace

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    PhotonDevice dev(nullptr);

    array_of_handles_release_frees_elements(dev);
    unset_parameter_releases_handle(dev);
    unset_all_parameters_releases_handles(dev);
    pixel_samples_int32_is_used_exactly(dev);
  }
  Kokkos::finalize();

  if (failures > 0)
    return 1;

  std::printf("photon_anari_lifecycle_test: all lifecycle checks passed\n");
  return 0;
}
