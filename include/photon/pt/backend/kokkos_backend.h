#pragma once

#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/bvh/bvh.h"
#include "photon/pt/geom/triangle_mesh.h"

namespace photon::pt {

struct Scene;

struct KokkosBackend : RayBackend {
  void build_accel(const Scene &scene) override;
  void trace_closest(const RayBatch &rays, HitBatch &hits) override;
  void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded) override;
  const char *name() const override { return "kokkos"; }

private:
  TriangleMesh m_mesh;
  Bvh m_bvh;
};

}
