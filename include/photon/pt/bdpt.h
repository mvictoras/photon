#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/camera.h"
#include "photon/pt/disney_bsdf.h"
#include "photon/pt/light.h"
#include "photon/pt/math.h"
#include "photon/pt/material.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"
#include "photon/pt/scene.h"

namespace photon::pt {

enum class VertexType : u32 { Camera = 0, Light = 1, Surface = 2 };

struct PathVertex {
  Vec3 p;
  Vec3 n;
  Vec3 beta;
  f32 pdf_fwd{0.f};
  f32 pdf_rev{0.f};
  Material mat;
  Vec3 wo;
  VertexType type{VertexType::Surface};
  bool is_specular{false};
  bool on_light{false};
  f32 light_pdf{0.f};
};

KOKKOS_FUNCTION inline f32 convert_density(f32 pdf_solid, const PathVertex &a, const PathVertex &b)
{
  const Vec3 w = b.p - a.p;
  const f32 dist2 = dot(w, w);
  if (dist2 == 0.f) return 0.f;
  const f32 cos_theta = Kokkos::fabs(dot(b.n, w * (1.f / Kokkos::sqrt(dist2))));
  return pdf_solid * cos_theta / dist2;
}

KOKKOS_FUNCTION inline f32 mis_weight_power(f32 a, f32 b)
{
  return a * a / (a * a + b * b);
}

struct LightSubpath {
  PathVertex v[8];
  i32 len{0};
};

}
