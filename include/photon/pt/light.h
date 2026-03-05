#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

enum class LightType : u32 {
  Point = 0,
  Directional = 1,
  Spot = 2,
  Area = 3,
  Environment = 4
};

struct Light {
  LightType type{LightType::Point};
  Vec3 position{0.f, 0.f, 0.f};
  Vec3 direction{0.f, -1.f, 0.f};
  Vec3 color{1.f, 1.f, 1.f};
  f32 intensity{1.f};
  f32 spot_angle{0.5f};
  f32 spot_falloff{0.1f};
  u32 mesh_prim_begin{0};
  u32 mesh_prim_count{0};
  f32 area{0.f};
};

struct LightSample {
  Vec3 wi;
  Vec3 Li;
  f32 pdf{0.f};
  f32 dist{0.f};
  bool is_delta{false};
};

KOKKOS_FUNCTION inline f32 clamp(f32 x, f32 a, f32 b)
{
  return x < a ? a : (x > b ? b : x);
}

KOKKOS_FUNCTION inline f32 smoothstep(f32 edge0, f32 edge1, f32 x)
{
  const f32 t = clamp((x - edge0) / (edge1 - edge0), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

KOKKOS_FUNCTION inline LightSample sample_light(const Light &light, const Vec3 &hit_pos, Rng &)
{
  constexpr f32 INF = 1e30f;

  if (light.type == LightType::Point) {
    const Vec3 to = light.position - hit_pos;
    const f32 dist = length(to);
    const Vec3 wi = dist > 0.f ? (to * (1.f / dist)) : Vec3{0.f, 1.f, 0.f};
    const f32 invDist2 = dist > 0.f ? (1.f / (dist * dist)) : 0.f;
    return {wi, light.color * (light.intensity * invDist2), 1.f, dist, true};
  }

  if (light.type == LightType::Directional) {
    const Vec3 wi = -normalize(light.direction);
    return {wi, light.color * light.intensity, 1.f, INF, true};
  }

  if (light.type == LightType::Spot) {
    const Vec3 to = light.position - hit_pos;
    const f32 dist = length(to);
    const Vec3 wi = dist > 0.f ? (to * (1.f / dist)) : Vec3{0.f, 1.f, 0.f};

    const Vec3 spotDir = normalize(light.direction);
    const Vec3 fromLight = -wi;
    const f32 cosAngle = dot(fromLight, spotDir);
    const f32 cosInner = Kokkos::cos(light.spot_angle - light.spot_falloff);
    const f32 cosOuter = Kokkos::cos(light.spot_angle);

    f32 falloff = 0.f;
    if (cosAngle >= cosInner)
      falloff = 1.f;
    else if (cosAngle <= cosOuter)
      falloff = 0.f;
    else {
      falloff = 1.f - smoothstep(cosOuter, cosInner, cosAngle);
    }

    const f32 invDist2 = dist > 0.f ? (1.f / (dist * dist)) : 0.f;
    return {wi, light.color * (light.intensity * invDist2 * falloff), 1.f, dist, true};
  }

  if (light.type == LightType::Area) {
    return {Vec3{0.f, 1.f, 0.f}, Vec3{0.f, 0.f, 0.f}, 0.f, 0.f, false};
  }

  return {Vec3{0.f, 1.f, 0.f}, Vec3{0.f, 0.f, 0.f}, 0.f, 0.f, false};
}

KOKKOS_FUNCTION inline f32 light_pdf(const Light &light, const Vec3 &, const Vec3 &)
{
  if (light.type == LightType::Point || light.type == LightType::Directional || light.type == LightType::Spot)
    return 0.f;
  return 0.f;
}

}
