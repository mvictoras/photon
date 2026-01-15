#pragma once

#include "opencode/pt/math.h"
#include "opencode/pt/ray.h"

namespace opencode::pt {

struct Hit
{
  f32 t{0.f};
  Vec3 p;
  Vec3 n;
  bool hit{false};
};

struct Sphere
{
  Vec3 c;
  f32 r{1.f};
  Vec3 albedo{1.f, 1.f, 1.f};

  KOKKOS_FUNCTION Hit intersect(const Ray &ray, f32 tmin, f32 tmax) const
  {
    const Vec3 oc = ray.org - c;
    const f32 a = dot(ray.dir, ray.dir);
    const f32 half_b = dot(oc, ray.dir);
    const f32 cc = dot(oc, oc) - r * r;
    const f32 disc = half_b * half_b - a * cc;
    if (disc < 0.f)
      return {};

    const f32 sdisc = Kokkos::sqrt(disc);

    f32 root = (-half_b - sdisc) / a;
    if (root < tmin || root > tmax) {
      root = (-half_b + sdisc) / a;
      if (root < tmin || root > tmax)
        return {};
    }

    Hit h;
    h.t = root;
    h.p = ray.at(root);
    h.n = normalize(h.p - c);
    h.hit = true;
    return h;
  }
};

} // namespace opencode::pt
