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

  Kokkos::View<Vec3 *, Kokkos::HostSpace> m_positions_h;
  Kokkos::View<u32 *, Kokkos::HostSpace> m_indices_h;
  Kokkos::View<Vec3 *, Kokkos::HostSpace> m_normals_h;
  Kokkos::View<Vec2 *, Kokkos::HostSpace> m_texcoords_h;
  Kokkos::View<u32 *, Kokkos::HostSpace> m_material_ids_h;
  Kokkos::View<Vec3 *, Kokkos::HostSpace> m_vertex_colors_h;
  bool m_has_normals{false};
  bool m_has_texcoords{false};
  bool m_has_material_ids{false};
  bool m_has_vertex_colors{false};

  Kokkos::View<Vec3 *, Kokkos::HostSpace> m_ray_org_h;
  Kokkos::View<Vec3 *, Kokkos::HostSpace> m_ray_dir_h;
  Kokkos::View<f32 *, Kokkos::HostSpace> m_ray_tmin_h;
  Kokkos::View<f32 *, Kokkos::HostSpace> m_ray_tmax_h;
  Kokkos::View<HitResult *, Kokkos::HostSpace> m_hit_h;
  Kokkos::View<u32 *, Kokkos::HostSpace> m_occ_h;
  u32 m_buf_size{0};

  void ensure_buffers(u32 n);
};

} // namespace photon::pt
#endif // PHOTON_HAS_EMBREE
