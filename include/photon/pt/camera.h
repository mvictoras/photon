#pragma once

#include "photon/pt/math.h"
#include "photon/pt/ray.h"

namespace photon::pt {

struct Camera
{
  Vec3 origin;
  Vec3 lower_left;
  Vec3 horizontal;
  Vec3 vertical;

  KOKKOS_FUNCTION static Camera make_pinhole(
      const Vec3 &look_from, const Vec3 &look_at, const Vec3 &vup, f32 vfov_degrees, f32 aspect)
  {
    const f32 theta = vfov_degrees * 3.1415926535f / 180.f;
    const f32 h = Kokkos::tan(theta * 0.5f);
    const f32 viewport_h = 2.f * h;
    const f32 viewport_w = aspect * viewport_h;

    const Vec3 w = normalize(look_from - look_at);
    const Vec3 u = normalize(cross(vup, w));
    const Vec3 v = cross(w, u);

    Camera cam;
    cam.origin = look_from;
    cam.horizontal = u * viewport_w;
    cam.vertical = v * viewport_h;
    cam.lower_left = cam.origin - cam.horizontal * 0.5f - cam.vertical * 0.5f - w;
    return cam;
  }

  KOKKOS_FUNCTION Ray ray(f32 s, f32 t) const
  {
    return Ray{origin, normalize(lower_left + horizontal * s + vertical * t - origin)};
  }
};

} // namespace photon::pt
