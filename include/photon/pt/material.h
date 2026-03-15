#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/math.h"

namespace photon::pt {

using i32 = int;

struct Material {
  Vec3 base_color{0.8f, 0.8f, 0.8f};
  f32 metallic{0.f};
  f32 roughness{0.5f};
  f32 ior{1.5f};
  f32 transmission{0.f};
  f32 specular{0.5f};
  f32 clearcoat{0.f};
  f32 clearcoat_roughness{0.03f};
  f32 sheen{0.f};
  f32 sheen_tint{0.5f};
  f32 specular_tint{0.f};
  f32 anisotropic{0.f};
  Vec3 emission{0.f, 0.f, 0.f};
  f32 emission_strength{0.f};
  f32 subsurface{0.f};
  Vec3 subsurface_color{1.f, 1.f, 1.f};
  f32 subsurface_radius{1.f};
  f32 alpha{1.f};
  bool thin{false};
  i32 base_color_tex{-1};
  i32 normal_tex{-1};
  i32 roughness_tex{-1};
  i32 metallic_tex{-1};
  i32 emission_tex{-1};
  i32 alpha_tex{-1};
};

KOKKOS_FUNCTION inline Vec3 material_emission(const Material &mat)
{
  return mat.emission * mat.emission_strength;
}

KOKKOS_FUNCTION inline bool material_is_emissive(const Material &mat)
{
  return mat.emission_strength > 0.f &&
      (mat.emission.x > 0.f || mat.emission.y > 0.f || mat.emission.z > 0.f);
}

}
