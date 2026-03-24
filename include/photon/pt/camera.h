#pragma once

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/ray.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

struct Camera {
  Vec3 origin;
  Vec3 lower_left;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 u, v, w;
  f32 lens_radius{0.f};
  f32 focus_dist{1.f};
  bool is_ortho{false};

  KOKKOS_FUNCTION static Camera make_pinhole(
      const Vec3 &look_from, const Vec3 &look_at, const Vec3 &vup, f32 vfov_degrees, f32 aspect)
  {
    return make_perspective(look_from, look_at, vup, vfov_degrees, aspect, 0.f, 1.f);
  }

  KOKKOS_FUNCTION static Camera make_perspective(
      const Vec3 &look_from, const Vec3 &look_at, const Vec3 &vup, f32 vfov_degrees, f32 aspect,
      f32 aperture = 0.f, f32 focus_dist = 1.f)
  {
    const f32 theta = vfov_degrees * 3.1415926535f / 180.f;
    const f32 h = Kokkos::tan(theta * 0.5f);
    const f32 viewport_h = 2.f * h;
    const f32 viewport_w = aspect * viewport_h;

    Camera cam{};
    cam.origin = look_from;
    cam.w = normalize(look_from - look_at);
    cam.u = normalize(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);

    cam.focus_dist = focus_dist;
    cam.lens_radius = aperture * 0.5f;
    cam.is_ortho = false;

    cam.horizontal = cam.u * (viewport_w * focus_dist);
    cam.vertical = cam.v * (viewport_h * focus_dist);
    cam.lower_left = cam.origin - cam.horizontal * 0.5f - cam.vertical * 0.5f - cam.w * focus_dist;
    return cam;
  }

  KOKKOS_FUNCTION static Camera make_orthographic(
      const Vec3 &look_from, const Vec3 &look_at, const Vec3 &vup, f32 ortho_height, f32 aspect)
  {
    Camera cam{};
    cam.origin = look_from;
    cam.w = normalize(look_from - look_at);
    cam.u = normalize(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);

    cam.focus_dist = 1.f;
    cam.lens_radius = 0.f;
    cam.is_ortho = true;

    const f32 viewport_h = ortho_height;
    const f32 viewport_w = ortho_height * aspect;

    cam.horizontal = cam.u * viewport_w;
    cam.vertical = cam.v * viewport_h;
    cam.lower_left = cam.origin - cam.horizontal * 0.5f - cam.vertical * 0.5f;
    return cam;
  }

  KOKKOS_FUNCTION Vec2 viewport_size() const
  {
    return {length(horizontal), length(vertical)};
  }

  KOKKOS_FUNCTION f32 importance(const Vec3 &dir) const
  {
    const f32 cos_theta = Kokkos::fabs(dot(dir, w * -1.f));
    if (cos_theta <= 0.f) return 0.f;
    const f32 cos2 = cos_theta * cos_theta;
    const f32 lens_area = lens_radius > 0.f ? PI * lens_radius * lens_radius : 1.f;
    const f32 viewport_area = length(horizontal) * length(vertical);
    return 1.f / (lens_area * viewport_area * cos2 * cos2);
  }

  KOKKOS_FUNCTION bool world_to_raster(const Vec3 &world_pos, int width, int height,
                                        f32 &raster_x, f32 &raster_y) const
  {
    if (is_ortho) return false;

    Vec3 view_dir = normalize(world_pos - origin);
    f32 cos_theta = dot(view_dir, w * -1.f);
    if (cos_theta <= 0.f) return false;

    Vec3 focus_point = origin + view_dir * (focus_dist / cos_theta);
    Vec3 p = focus_point - lower_left;
    f32 s = dot(p, normalize(horizontal)) / length(horizontal);
    f32 t = dot(p, normalize(vertical)) / length(vertical);
    if (s < 0.f || s > 1.f || t < 0.f || t > 1.f) return false;
    raster_x = s * f32(width);
    raster_y = (1.f - t) * f32(height);
    return true;
  }

  KOKKOS_FUNCTION Ray ray(f32 s, f32 t) const
  {
    Rng rng(1u);
    return ray(s, t, rng);
  }

  KOKKOS_FUNCTION Ray ray(f32 s, f32 t, Rng &rng) const
  {
    if (is_ortho) {
      const Vec3 origin_offset = lower_left + horizontal * s + vertical * t;
      return Ray{origin_offset, -w};
    }

    if (lens_radius <= 0.f) {
      return Ray{origin, normalize(lower_left + horizontal * s + vertical * t - origin)};
    }

    const Vec2 rd = sample_concentric_disk(rng) * lens_radius;
    const Vec3 offset = u * rd.x + v * rd.y;
    const Vec3 target = lower_left + horizontal * s + vertical * t;
    return Ray{origin + offset, normalize(target - origin - offset)};
  }
};

} // namespace photon::pt
