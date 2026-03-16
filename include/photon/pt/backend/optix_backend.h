#pragma once
#ifdef PHOTON_HAS_OPTIX

#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/geom/triangle_mesh.h"

#include <cuda_runtime.h>
#include <optix.h>

namespace photon::pt {

struct OptixBackend : RayBackend {
  OptixBackend();
  ~OptixBackend() override;

  void build_accel(const Scene &scene) override;
  void trace_closest(const RayBatch &rays, HitBatch &hits) override;
  void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded) override;
  const char *name() const override { return "optix"; }

private:
  OptixDeviceContext m_context{nullptr};
  OptixTraversableHandle m_gas_handle{0};
  CUdeviceptr m_gas_buffer{0};
  OptixModule m_module{nullptr};
  OptixPipeline m_pipeline{nullptr};
  OptixShaderBindingTable m_sbt{};
  CUdeviceptr m_d_params{0};
  TriangleMesh m_mesh;

  void create_pipeline();
};

}
#endif
