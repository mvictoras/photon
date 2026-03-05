#pragma once
#ifdef PHOTON_HAS_HIPRT

#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/geom/triangle_mesh.h"

#include <hiprt/hiprt.h>

namespace photon::pt {

struct HiprtBackend : RayBackend {
  HiprtBackend();
  ~HiprtBackend() override;

  void build_accel(const Scene &scene) override;
  void trace_closest(const RayBatch &rays, HitBatch &hits) override;
  void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded) override;
  const char *name() const override { return "hiprt"; }

private:
  hiprtContext m_context{nullptr};
  hiprtGeometry m_geometry{nullptr};
  hiprtScene m_scene_handle{nullptr};
  TriangleMesh m_mesh;
};

}
#endif
