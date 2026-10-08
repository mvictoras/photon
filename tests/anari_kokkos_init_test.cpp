// Acceptance tests for issue #11: safe Kokkos initialization.
//
// PhotonDevice must initialize Kokkos itself when the host process has not
// (e.g. ParaView loading the ANARI library), and must do so at most once per
// process no matter how many PhotonDevice instances are created. Destroying
// a device must not finalize Kokkos while views (accumulation buffers) may
// still be live.
//
// The CUDA-specific part — targeting the host's already-selected CUDA device
// via cudaGetDevice() — is guarded by KOKKOS_ENABLE_CUDA and cannot be
// exercised on the CPU-only test build; this test pins the process-level
// guarantees that hold on every build.
//
// Failures are counted, not asserted: the Release build defines NDEBUG, which
// would compile plain assert() out and make every check here vacuous.

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

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

// Prove Kokkos is not just flagged initialized but actually usable.
void kokkos_works()
{
  Kokkos::View<float *> view("issue11_probe", 16);
  Kokkos::deep_copy(view, 1.0f);
  auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  expect(mirror(0) == 1.0f,
      "Kokkos view round-trip failed after device-driven initialization");
}

} // namespace

int main()
{
  // NOTE: the host process deliberately does NOT call Kokkos::initialize()
  // here — the first PhotonDevice must do it.

  {
    PhotonDevice first(nullptr);
    expect(Kokkos::is_initialized(),
        "PhotonDevice did not initialize Kokkos for the host process");
    kokkos_works();

    // A second instance must not re-initialize (Kokkos aborts on a redundant
    // initialize(), so reaching here at all means "at most once" held).
    {
      PhotonDevice second(nullptr);
      kokkos_works();
    }

    // Destruction must not finalize Kokkos out from under remaining users.
    expect(Kokkos::is_initialized(),
        "destroying a PhotonDevice finalized Kokkos while views may be live");
    kokkos_works();

    PhotonDevice third(nullptr);
    kokkos_works();
  }

  expect(Kokkos::is_initialized(),
      "the last destroyed PhotonDevice finalized Kokkos prematurely");

  Kokkos::finalize();

  if (failures > 0)
    return 1;

  std::printf("photon_anari_kokkos_init_test: all init checks passed\n");
  return 0;
}
