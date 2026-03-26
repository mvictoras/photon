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

  // Device-side copies of mesh attributes for non-CUDA Kokkos backends.
  void *m_d_mesh_positions{nullptr};
  void *m_d_mesh_indices{nullptr};
  void *m_d_mesh_normals{nullptr};
  void *m_d_mesh_texcoords{nullptr};
  void *m_d_mesh_material_ids{nullptr};
  void *m_d_mesh_vertex_colors{nullptr};

  // Persistent staging buffers for ray/hit data (reused across trace calls).
  void *m_d_ray_origins{nullptr};
  void *m_d_ray_directions{nullptr};
  void *m_d_ray_tmin{nullptr};
  void *m_d_ray_tmax{nullptr};
  void *m_d_hit_results{nullptr};
  void *m_d_occluded{nullptr};
  size_t m_staging_capacity{0};  // number of rays the buffers can hold

  void create_pipeline();
  void free_device_mesh_buffers();
  void ensure_staging_buffers(u32 count);
  void free_staging_buffers();
};

}
#endif
