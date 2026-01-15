#pragma once

#include "opencode/pt/math.h"
#include "opencode/pt/ray.h"

namespace opencode::pt {

struct Aabb
{
  Vec3 lo{+1e30f, +1e30f, +1e30f};
  Vec3 hi{-1e30f, -1e30f, -1e30f};

  KOKKOS_FUNCTION void extend(const Vec3 &p)
  {
    lo.x = lo.x < p.x ? lo.x : p.x;
    lo.y = lo.y < p.y ? lo.y : p.y;
    lo.z = lo.z < p.z ? lo.z : p.z;

    hi.x = hi.x > p.x ? hi.x : p.x;
    hi.y = hi.y > p.y ? hi.y : p.y;
    hi.z = hi.z > p.z ? hi.z : p.z;
  }

  KOKKOS_FUNCTION void extend(const Aabb &b)
  {
    extend(b.lo);
    extend(b.hi);
  }
};

KOKKOS_FUNCTION inline bool hit_aabb(const Aabb &box, const Ray &r, f32 tmin, f32 tmax)
{
  for (int a = 0; a < 3; ++a) {
    const f32 ro = a == 0 ? r.org.x : (a == 1 ? r.org.y : r.org.z);
    const f32 rd = a == 0 ? r.dir.x : (a == 1 ? r.dir.y : r.dir.z);
    const f32 invd = 1.f / rd;

    const f32 t0 = ((a == 0 ? box.lo.x : (a == 1 ? box.lo.y : box.lo.z)) - ro) * invd;
    const f32 t1 = ((a == 0 ? box.hi.x : (a == 1 ? box.hi.y : box.hi.z)) - ro) * invd;

    const f32 tnear = t0 < t1 ? t0 : t1;
    const f32 tfar = t0 < t1 ? t1 : t0;

    tmin = tnear > tmin ? tnear : tmin;
    tmax = tfar < tmax ? tfar : tmax;
    if (tmax <= tmin)
      return false;
  }

  return true;
}

} // namespace opencode::pt
