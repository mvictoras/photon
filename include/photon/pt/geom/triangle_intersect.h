#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/ray.h"

namespace photon::pt {

struct TriHit
{
  f32 t{0.f};
  f32 u{0.f};
  f32 v{0.f};
  bool hit{false};
};

KOKKOS_FUNCTION inline TriHit intersect_triangle(const Ray &ray, const Vec3 &v0, const Vec3 &v1, const Vec3 &v2,
    f32 tmin, f32 tmax)
{
  const Vec3 e1 = v1 - v0;
  const Vec3 e2 = v2 - v0;
  const Vec3 p = cross(ray.dir, e2);
  const f32 det = dot(e1, p);

  if (Kokkos::fabs(det) < 1e-8f)
    return {};

  const f32 inv_det = 1.f / det;
  const Vec3 tvec = ray.org - v0;
  const f32 u = dot(tvec, p) * inv_det;
  if (u < 0.f || u > 1.f)
    return {};

  const Vec3 q = cross(tvec, e1);
  const f32 v = dot(ray.dir, q) * inv_det;
  if (v < 0.f || (u + v) > 1.f)
    return {};

  const f32 t = dot(e2, q) * inv_det;
  if (t < tmin || t > tmax)
    return {};

  TriHit h;
  h.t = t;
  h.u = u;
  h.v = v;
  h.hit = true;
  return h;
}

} // namespace photon::pt
