// Verifies object lifetime handling in PhotonDevice:
//   1. releasing an array-of-handles releases the handles it contains
//   2. unsetParameter releases the handle stored in the parameter slot
//   3. unsetAllParameters releases every handle parameter
// Regression tests for the refcount leaks that leaked surfaces/geometries
// when hosts released arrays or cleared parameters.

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#include <anari/anari.h>

#include <cstdint>
#include <cstdio>

using photon::anari_device::PhotonDevice;

namespace {

int failures = 0;

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
  Kokkos::initialize();
  {
    PhotonDevice dev(nullptr);

    // --- 1. Array of handles: releasing the array releases its contents ---
    ANARISurface surface = dev.newSurface();
    const uintptr_t handles[2] = {uintptr_t(surface), 0};
    ANARIArray1D arr = dev.newArray1D(handles, nullptr, nullptr, ANARI_SURFACE, 2);

    // The array retained the surface (the app still holds its own reference).
    dev.release((ANARIObject)arr);
    expect(dev.getObject(uintptr_t(surface)) != nullptr,
        "surface freed while array still holds a reference to it");

    // Last reference released — the contained surface must now be freed.
    dev.release((ANARIObject)surface);
    expect(dev.getObject(uintptr_t(surface)) == nullptr,
        "surface leaked after array-of-handles was released");

    // --- 2. unsetParameter releases the handle in the parameter slot ---
    ANARISurface s2 = dev.newSurface();
    ANARIWorld world = dev.newWorld();
    dev.setParameter((ANARIObject)world, "surface", ANARI_SURFACE, &s2);
    expect(dev.getObject(uintptr_t(s2)) != nullptr
            && dev.getObject(uintptr_t(s2))->refcount == 2,
        "setParameter did not retain the handle parameter");

    dev.unsetParameter((ANARIObject)world, "surface");
    expect(dev.getObject(uintptr_t(s2)) != nullptr,
        "surface freed while app reference still exists after unsetParameter");
    dev.release((ANARIObject)s2);
    expect(dev.getObject(uintptr_t(s2)) == nullptr,
        "surface leaked after unsetParameter released the parameter slot");

    // --- 3. unsetAllParameters releases every handle parameter ---
    ANARISurface s3 = dev.newSurface();
    dev.setParameter((ANARIObject)world, "surface", ANARI_SURFACE, &s3);
    expect(dev.getObject(uintptr_t(s3)) != nullptr
            && dev.getObject(uintptr_t(s3))->refcount == 2,
        "setParameter did not retain the handle parameter");

    dev.unsetAllParameters((ANARIObject)world);
    dev.release((ANARIObject)s3);
    expect(dev.getObject(uintptr_t(s3)) == nullptr,
        "surface leaked after unsetAllParameters cleared handle parameters");

    // --- 4. Array created empty and filled via mapArray does NOT own its
    // contents: releasing the array must not release the contained handle
    // (it was never retained by the device).
    ANARISurface s4 = dev.newSurface();
    ANARIArray1D arr2 =
        dev.newArray1D(nullptr, nullptr, nullptr, ANARI_SURFACE, 1);
    auto *mem = static_cast<uintptr_t *>(dev.mapArray(arr2));
    mem[0] = uintptr_t(s4);
    dev.unmapArray(arr2);

    dev.release((ANARIObject)arr2);
    expect(dev.getObject(uintptr_t(s4)) != nullptr,
        "device released a handle it never owned (mapArray-filled array)");
    dev.release((ANARIObject)s4);
    expect(dev.getObject(uintptr_t(s4)) == nullptr,
        "surface not freed after its last reference was released");
  }
  Kokkos::finalize();

  if (failures > 0)
    return 1;

  std::printf("photon_anari_refcount_test: all lifetime checks passed\n");
  return 0;
}
