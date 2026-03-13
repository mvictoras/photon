#include "photon/pt/backend/kokkos_backend.h"

#include "photon/pt/scene.h"

#include "photon/pt/geom/mesh_intersector.h"
#include "photon/pt/ray.h"

namespace photon::pt {

void KokkosBackend::build_accel(const Scene &scene)
{
  m_mesh = scene.mesh;
  m_bvh = scene.bvh;

  if (m_bvh.nodes.extent(0) == 0 || m_bvh.prim_ids.extent(0) == 0) {
    m_bvh = Bvh::build_cpu(m_mesh);
  }
}

void KokkosBackend::trace_closest(const RayBatch &rays, HitBatch &hits)
{
  const auto mesh = m_mesh;
  const auto bvh = m_bvh;

  const auto hit_view = hits.hits;
  const auto orgs = rays.origins;
  const auto dirs = rays.directions;
  const auto tmins = rays.tmin;
  const auto tmaxs = rays.tmax;

  Kokkos::parallel_for(
      "trace_closest", rays.count, KOKKOS_LAMBDA(const u32 i) {
        Ray ray{orgs(i), dirs(i)};
        const MeshHit mh = intersect_mesh_bvh(mesh, bvh, ray, tmins(i), tmaxs(i));

        HitResult &hr = hit_view(i);
        hr.hit = mh.hit;
        if (mh.hit) {
          hr.t = mh.t;
          hr.position = mh.p;
          hr.normal = mh.n;
          hr.shading_normal = mh.shading_normal;
          hr.uv = mh.uv;
          hr.prim_id = mh.prim_id;
          hr.geom_id = mh.geom_id;
          hr.material_id = mh.material_id;
          hr.interpolated_color = mh.interpolated_color;
          hr.has_interpolated_color = mh.has_interpolated_color;
        }
      });
  Kokkos::fence();
}

void KokkosBackend::trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded)
{
  const auto mesh = m_mesh;
  const auto bvh = m_bvh;

  const auto orgs = rays.origins;
  const auto dirs = rays.directions;
  const auto tmins = rays.tmin;
  const auto tmaxs = rays.tmax;

  Kokkos::parallel_for(
      "trace_occluded", rays.count, KOKKOS_LAMBDA(const u32 i) {
        Ray ray{orgs(i), dirs(i)};
        const MeshHit mh = intersect_mesh_bvh(mesh, bvh, ray, tmins(i), tmaxs(i));
        occluded(i) = mh.hit ? 1u : 0u;
      });
  Kokkos::fence();
}

}
