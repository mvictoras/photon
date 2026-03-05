#pragma once
#ifdef PHOTON_HAS_EMBREE

#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/geom/triangle_mesh.h"

#include <embree4/rtcore.h>

namespace photon::pt {

struct EmbreeBackend : RayBackend {
  EmbreeBackend();
  ~EmbreeBackend() override;

  void build_accel(const Scene &scene) override;
  void trace_closest(const RayBatch &rays, HitBatch &hits) override;
  void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded) override;
  const char *name() const override { return "embree"; }

private:
  RTCDevice m_device{nullptr};
  RTCScene m_scene{nullptr};
  TriangleMesh m_mesh;
};

} // namespace photon::pt
#endif // PHOTON_HAS_EMBREE
