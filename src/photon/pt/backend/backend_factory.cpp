#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/backend/kokkos_backend.h"

#ifdef PHOTON_HAS_EMBREE
#include "photon/pt/backend/embree_backend.h"
#endif
#ifdef PHOTON_HAS_OPTIX
#include "photon/pt/backend/optix_backend.h"
#include <cuda_runtime.h>
#endif
#ifdef PHOTON_HAS_HIPRT
#include "photon/pt/backend/hiprt_backend.h"
#endif

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>

namespace photon::pt {

std::unique_ptr<RayBackend> create_best_backend()
{
  const char *env = std::getenv("PHOTON_BACKEND");

  if (env) {
    if (std::strcmp(env, "kokkos") == 0)
      return std::make_unique<KokkosBackend>();
#ifdef PHOTON_HAS_EMBREE
    if (std::strcmp(env, "embree") == 0)
      return std::make_unique<EmbreeBackend>();
#endif
#ifdef PHOTON_HAS_OPTIX
    if (std::strcmp(env, "optix") == 0)
      return std::make_unique<OptixBackend>();
#endif
#ifdef PHOTON_HAS_HIPRT
    if (std::strcmp(env, "hiprt") == 0)
      return std::make_unique<HiprtBackend>();
#endif
  }

#ifdef PHOTON_HAS_OPTIX
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
      try {
        auto backend = std::make_unique<OptixBackend>();
        std::fprintf(stderr, "[photon] Using OptiX backend (NVIDIA GPU detected)\n");
        return backend;
      } catch (const std::exception &e) {
        std::fprintf(stderr, "[photon] OptiX init failed (%s), falling back\n", e.what());
      }
    }
  }
#endif

#ifdef PHOTON_HAS_HIPRT
  std::fprintf(stderr, "[photon] Using HIP RT backend (AMD GPU detected)\n");
  return std::make_unique<HiprtBackend>();
#endif

#ifdef PHOTON_HAS_EMBREE
  std::fprintf(stderr, "[photon] Using Embree backend (CPU)\n");
  return std::make_unique<EmbreeBackend>();
#endif

  std::fprintf(stderr, "[photon] Using Kokkos fallback backend\n");
  return std::make_unique<KokkosBackend>();
}

}
