#ifdef PHOTON_HAS_EMBREE

#include "photon/pt/backend/embree_backend.h"

#include "photon/pt/geom/triangle_intersect.h"
#include "photon/pt/scene.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace photon::pt {
namespace {

void embree_error_handler(void *, const RTCError code, const char *str)
{
  std::fprintf(stderr, "[photon][embree] error %d: %s\n", int(code), str ? str : "(null)");
}

inline Vec3 normalize_safe(const Vec3 v)
{
  const f32 len2 = dot(v, v);
  if (len2 <= 0.f)
    return {0.f, 0.f, 0.f};
  const f32 inv_len = 1.f / std::sqrt(len2);
  return v * inv_len;
}

inline void compute_shading(const TriangleMesh &mesh,
                            const u32 prim_id,
                            const f32 u,
                            const f32 v,
                            Vec3 &out_ng,
                            Vec3 &out_ns,
                            Vec2 &out_uv,
                            u32 &out_material_id)
{
  const u32 i0 = mesh.indices(prim_id * 3 + 0);
  const u32 i1 = mesh.indices(prim_id * 3 + 1);
  const u32 i2 = mesh.indices(prim_id * 3 + 2);

  const Vec3 p0 = mesh.positions(i0);
  const Vec3 p1 = mesh.positions(i1);
  const Vec3 p2 = mesh.positions(i2);

  const f32 w = 1.f - u - v;

  const Vec3 ng = normalize_safe(cross(p1 - p0, p2 - p0));
  out_ng = ng;

  if (mesh.has_normals()) {
    const Vec3 n0 = mesh.normals(i0);
    const Vec3 n1 = mesh.normals(i1);
    const Vec3 n2 = mesh.normals(i2);
    out_ns = normalize_safe(n0 * w + n1 * u + n2 * v);
  } else {
    out_ns = ng;
  }

  if (mesh.has_texcoords()) {
    const Vec2 t0 = mesh.texcoords(i0);
    const Vec2 t1 = mesh.texcoords(i1);
    const Vec2 t2 = mesh.texcoords(i2);
    out_uv = {t0.x * w + t1.x * u + t2.x * v, t0.y * w + t1.y * u + t2.y * v};
  } else {
    out_uv = {u, v};
  }

  if (mesh.has_material_ids()) {
    out_material_id = mesh.material_ids(prim_id);
  } else {
    out_material_id = 0u;
  }
}

}

EmbreeBackend::EmbreeBackend()
{
  m_device = rtcNewDevice(nullptr);
  if (!m_device) {
    throw std::runtime_error("rtcNewDevice failed");
  }
  rtcSetDeviceErrorFunction(m_device, embree_error_handler, nullptr);
}

EmbreeBackend::~EmbreeBackend()
{
  if (m_scene) {
    rtcReleaseScene(m_scene);
    m_scene = nullptr;
  }
  if (m_device) {
    rtcReleaseDevice(m_device);
    m_device = nullptr;
  }
}

void EmbreeBackend::build_accel(const Scene &scene)
{
  m_mesh = scene.mesh;

  if (m_scene) {
    rtcReleaseScene(m_scene);
    m_scene = nullptr;
  }

  m_scene = rtcNewScene(m_device);

  RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.positions);
  auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.indices);

  const size_t vertex_count = pos_h.extent(0);
  const size_t index_count = idx_h.extent(0);

  auto *vb = static_cast<float *>(rtcSetNewGeometryBuffer(
      geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3, vertex_count));

  for (size_t i = 0; i < vertex_count; ++i) {
    vb[i * 3 + 0] = pos_h(i).x;
    vb[i * 3 + 1] = pos_h(i).y;
    vb[i * 3 + 2] = pos_h(i).z;
  }

  auto *ib = static_cast<unsigned *>(rtcSetNewGeometryBuffer(
      geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned) * 3, index_count / 3));

  for (size_t i = 0; i < index_count / 3; ++i) {
    ib[i * 3 + 0] = idx_h(i * 3 + 0);
    ib[i * 3 + 1] = idx_h(i * 3 + 1);
    ib[i * 3 + 2] = idx_h(i * 3 + 2);
  }

  rtcCommitGeometry(geom);
  rtcAttachGeometry(m_scene, geom);
  rtcReleaseGeometry(geom);

  rtcCommitScene(m_scene);
}

void EmbreeBackend::trace_closest(const RayBatch &rays, HitBatch &hits)
{
  if (!m_scene) {
    throw std::runtime_error("EmbreeBackend::trace_closest called before build_accel");
  }

  const u32 n = rays.count;
  if (hits.count != n) {
    throw std::runtime_error("HitBatch.count must match RayBatch.count");
  }

  auto org_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.origins);
  auto dir_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.directions);
  auto tmin_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.tmin);
  auto tmax_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.tmax);

  auto hit_h = Kokkos::create_mirror_view(hits.hits);

  RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  for (u32 i = 0; i < n; ++i) {
    RTCRayHit rh{};
    rh.ray.org_x = org_h(i).x;
    rh.ray.org_y = org_h(i).y;
    rh.ray.org_z = org_h(i).z;
    rh.ray.dir_x = dir_h(i).x;
    rh.ray.dir_y = dir_h(i).y;
    rh.ray.dir_z = dir_h(i).z;
    rh.ray.tnear = tmin_h(i);
    rh.ray.tfar = tmax_h(i);
    rh.ray.mask = 0xFFFFFFFFu;
    rh.ray.flags = 0;

    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.primID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(m_scene, &context, &rh);

    HitResult hr{};
    hr.hit = (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID);
    if (hr.hit) {
      hr.t = rh.ray.tfar;
      hr.position = org_h(i) + dir_h(i) * hr.t;
      hr.prim_id = rh.hit.primID;
      hr.geom_id = rh.hit.geomID;
      hr.inst_id = 0u;

      Vec3 ng, ns;
      Vec2 uv;
      u32 mat_id;
      compute_shading(m_mesh, hr.prim_id, rh.hit.u, rh.hit.v, ng, ns, uv, mat_id);
      hr.normal = ng;
      hr.shading_normal = ns;
      hr.uv = uv;
      hr.material_id = mat_id;
    }

    hit_h(i) = hr;
  }

  Kokkos::deep_copy(hits.hits, hit_h);
}

void EmbreeBackend::trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded)
{
  if (!m_scene) {
    throw std::runtime_error("EmbreeBackend::trace_occluded called before build_accel");
  }

  const u32 n = rays.count;
  auto org_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.origins);
  auto dir_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.directions);
  auto tmin_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.tmin);
  auto tmax_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.tmax);

  auto occ_h = Kokkos::create_mirror_view(occluded);

  RTCOccludedContext context;
  rtcInitOccludedContext(&context);

  for (u32 i = 0; i < n; ++i) {
    RTCRay ray{};
    ray.org_x = org_h(i).x;
    ray.org_y = org_h(i).y;
    ray.org_z = org_h(i).z;
    ray.dir_x = dir_h(i).x;
    ray.dir_y = dir_h(i).y;
    ray.dir_z = dir_h(i).z;
    ray.tnear = tmin_h(i);
    ray.tfar = tmax_h(i);
    ray.mask = 0xFFFFFFFFu;
    ray.flags = 0;

    rtcOccluded1(m_scene, &context, &ray);
    occ_h(i) = (ray.tfar < 0.f) ? 1u : 0u;
  }

  Kokkos::deep_copy(occluded, occ_h);
}

}

#endif
