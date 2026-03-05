#include "photon/pt/backend/kokkos_backend.h"
#include "photon/pt/backend/ray_backend.h"

#include <cstdlib>
#include <cstring>

namespace photon::pt {

std::unique_ptr<RayBackend> create_best_backend()
{
  const char *env = std::getenv("PHOTON_BACKEND");
  if (env) {
    if (std::strcmp(env, "kokkos") == 0)
      return std::make_unique<KokkosBackend>();
  }

  return std::make_unique<KokkosBackend>();
}

}
