#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/material.h"
#include "photon/pt/math.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

struct BsdfSample {
  Vec3 wi;
  Vec3 f;
  f32 pdf{0.f};
  bool is_specular{false};
};

KOKKOS_FUNCTION inline f32 saturate(f32 x)
{
  return Kokkos::fmin(1.f, Kokkos::fmax(0.f, x));
}

KOKKOS_FUNCTION inline f32 dielectric_f0(f32 ior)
{
  const f32 a = (1.f - ior) / (1.f + ior);
  return a * a;
}

KOKKOS_FUNCTION inline Vec3 fresnel_schlick(f32 cosTheta, Vec3 F0)
{
  const f32 m = 1.f - saturate(cosTheta);
  const f32 m2 = m * m;
  const f32 m5 = m2 * m2 * m;
  return F0 + (Vec3{1.f, 1.f, 1.f} - F0) * m5;
}

KOKKOS_FUNCTION inline f32 D_ggx(f32 NoH, f32 a)
{
  const f32 a2 = a * a;
  const f32 d = NoH * NoH * (a2 - 1.f) + 1.f;
  return a2 / (PI * d * d);
}

KOKKOS_FUNCTION inline f32 G1_smith(f32 NoV, f32 a)
{
  const f32 a2 = a * a;
  const f32 denom = NoV + Kokkos::sqrt(a2 + (1.f - a2) * NoV * NoV);
  if (denom <= 0.f)
    return 0.f;
  return (2.f * NoV) / denom;
}

KOKKOS_FUNCTION inline f32 G_smith(f32 NoV, f32 NoL, f32 a)
{
  return G1_smith(NoV, a) * G1_smith(NoL, a);
}

KOKKOS_FUNCTION inline f32 ggx_pdf_reflection(f32 NoH, f32 VoH, f32 a)
{
  const f32 D = D_ggx(NoH, a);
  const f32 denom = 4.f * Kokkos::fmax(VoH, 1e-7f);
  return D * NoH / denom;
}

KOKKOS_FUNCTION inline void disney_lobe_weights(const Material &mat, f32 &w_diffuse,
    f32 &w_specular, f32 &w_transmission, f32 &w_clearcoat)
{
  w_diffuse = (1.f - mat.metallic) * (1.f - mat.transmission);
  w_specular = 1.f;
  w_transmission = mat.transmission * (1.f - mat.metallic);
  w_clearcoat = mat.clearcoat * 0.25f;
}

KOKKOS_FUNCTION inline Vec3 eval_diffuse(const Material &mat, f32 NoL)
{
  const f32 w = (1.f - mat.metallic) * (1.f - mat.transmission);
  if (w <= 0.f)
    return Vec3{0.f, 0.f, 0.f};
  return mat.base_color * (w * INV_PI * NoL);
}

KOKKOS_FUNCTION inline Vec3 eval_specular_reflection(const Material &mat,
    const Vec3 &wo, const Vec3 &wi, const Vec3 &n, f32 a)
{
  const f32 NoV = saturate(dot(n, wo));
  const f32 NoL = saturate(dot(n, wi));
  if (NoV <= 0.f || NoL <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const Vec3 h = normalize(wi + wo);
  const f32 NoH = saturate(dot(n, h));
  const f32 VoH = saturate(dot(wo, h));

  const f32 f0d = dielectric_f0(mat.ior);
  Vec3 F0_dielectric = Vec3{f0d, f0d, f0d};
  if (mat.specular_tint > 0.f) {
    const f32 lum = 0.2126f * mat.base_color.x + 0.7152f * mat.base_color.y + 0.0722f * mat.base_color.z;
    const Vec3 c_tint = lum > 0.f ? mat.base_color * (1.f / lum) : Vec3{1.f, 1.f, 1.f};
    F0_dielectric = lerp(F0_dielectric, c_tint * f0d, mat.specular_tint);
  }
  Vec3 F0 = lerp(F0_dielectric, mat.base_color, mat.metallic);

  const Vec3 F = fresnel_schlick(VoH, F0);
  const f32 D = D_ggx(NoH, a);
  const f32 G = G_smith(NoV, NoL, a);

  const f32 denom = 4.f * Kokkos::fmax(NoV, 1e-7f);
  return F * (D * G * NoL / denom);
}

KOKKOS_FUNCTION inline Vec3 eval_sheen(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n)
{
  if (mat.sheen <= 0.f || mat.metallic >= 1.f)
    return Vec3{0.f, 0.f, 0.f};

  const Vec3 h = normalize(wi + wo);
  const f32 HoL = Kokkos::fabs(dot(h, wi));

  const f32 lum = 0.2126f * mat.base_color.x + 0.7152f * mat.base_color.y + 0.0722f * mat.base_color.z;
  const Vec3 c_tint = lum > 0.f ? mat.base_color * (1.f / lum) : Vec3{1.f, 1.f, 1.f};
  const Vec3 c_sheen = lerp(Vec3{1.f, 1.f, 1.f}, c_tint, mat.sheen_tint);

  const f32 m = 1.f - HoL;
  const f32 m2 = m * m;
  const f32 m5 = m2 * m2 * m;

  const f32 NoL = saturate(dot(n, wi));
  const f32 w = (1.f - mat.metallic) * (1.f - mat.transmission);
  return c_sheen * (mat.sheen * m5 * w * NoL);
}

KOKKOS_FUNCTION inline Vec3 eval_clearcoat(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n)
{
  if (mat.clearcoat <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 NoV = saturate(dot(n, wo));
  const f32 NoL = saturate(dot(n, wi));
  if (NoV <= 0.f || NoL <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 a = Kokkos::fmax(mat.clearcoat_roughness, 0.001f);

  const Vec3 h = normalize(wi + wo);
  const f32 NoH = saturate(dot(n, h));
  const f32 VoH = saturate(dot(wo, h));

  const f32 F0 = dielectric_f0(1.5f);
  const Vec3 F = fresnel_schlick(VoH, Vec3{F0, F0, F0});
  const f32 D = D_ggx(NoH, a);
  const f32 G = G_smith(NoV, NoL, a);

  const f32 denom = 4.f * Kokkos::fmax(NoV, 1e-7f);
  const f32 w = mat.clearcoat * 0.25f;
  return F * (w * D * G * NoL / denom);
}

KOKKOS_FUNCTION inline Vec3 disney_bsdf_eval(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n, const Vec3 &shading_n)
{
  if (dot(wo, n) <= 0.f || dot(wi, n) <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 NoL = saturate(dot(shading_n, wi));
  const f32 NoV = saturate(dot(shading_n, wo));
  if (NoL <= 0.f || NoV <= 0.f)
    return Vec3{0.f, 0.f, 0.f};

  const f32 a = Kokkos::fmax(mat.roughness, 0.001f);

  Vec3 f = eval_diffuse(mat, NoL);
  f += eval_specular_reflection(mat, wo, wi, shading_n, a);
  f += eval_sheen(mat, wo, wi, shading_n);
  f += eval_clearcoat(mat, wo, wi, shading_n);
  return f;
}

KOKKOS_FUNCTION inline f32 disney_bsdf_pdf(const Material &mat, const Vec3 &wo,
    const Vec3 &wi, const Vec3 &n, const Vec3 &shading_n)
{
  if (dot(wo, n) <= 0.f || dot(wi, n) <= 0.f)
    return 0.f;

  const f32 NoL = saturate(dot(shading_n, wi));
  const f32 NoV = saturate(dot(shading_n, wo));
  if (NoL <= 0.f || NoV <= 0.f)
    return 0.f;

  const f32 a = Kokkos::fmax(mat.roughness, 0.001f);

  f32 w_diffuse, w_specular, w_transmission, w_clearcoat;
  disney_lobe_weights(mat, w_diffuse, w_specular, w_transmission, w_clearcoat);

  const f32 sum_w = w_diffuse + w_specular + w_transmission + w_clearcoat;
  if (sum_w <= 0.f)
    return 0.f;

  const f32 p_diffuse = w_diffuse / sum_w;
  const f32 p_specular = w_specular / sum_w;
  const f32 p_transmission = w_transmission / sum_w;
  const f32 p_clearcoat = w_clearcoat / sum_w;

  f32 pdf = 0.f;

  if (p_diffuse > 0.f)
    pdf += p_diffuse * (NoL * INV_PI);

  if (p_specular > 0.f) {
    const Vec3 h = normalize(wi + wo);
    const f32 NoH = saturate(dot(shading_n, h));
    const f32 VoH = saturate(dot(wo, h));
    pdf += p_specular * ggx_pdf_reflection(NoH, VoH, a);
  }

  if (p_clearcoat > 0.f) {
    const Vec3 h = normalize(wi + wo);
    const f32 NoH = saturate(dot(shading_n, h));
    const f32 VoH = saturate(dot(wo, h));
    const f32 a_cc = Kokkos::fmax(mat.clearcoat_roughness, 0.001f);
    pdf += p_clearcoat * ggx_pdf_reflection(NoH, VoH, a_cc);
  }

  if (p_transmission > 0.f) {
    pdf += p_transmission * (NoL * INV_PI);
  }

  return Kokkos::fmax(pdf, 0.f);
}

KOKKOS_FUNCTION inline Vec3 sample_ggx_half_vector(const Vec3 &n, f32 a, Rng &rng)
{
  Vec3 tangent{}, bitangent{};
  onb_from_normal(n, tangent, bitangent);

  const f32 r1 = rng.next_f32();
  const f32 r2 = rng.next_f32();
  const f32 a2 = a * a;

  const f32 cos_theta = Kokkos::sqrt((1.f - r1) / (1.f + (a2 - 1.f) * r1));
  const f32 sin_theta = Kokkos::sqrt(Kokkos::fmax(0.f, 1.f - cos_theta * cos_theta));
  const f32 phi = 2.f * PI * r2;

  const f32 x = sin_theta * Kokkos::cos(phi);
  const f32 y = sin_theta * Kokkos::sin(phi);
  const f32 z = cos_theta;

  return normalize(tangent * x + bitangent * y + n * z);
}

KOKKOS_FUNCTION inline BsdfSample disney_bsdf_sample(const Material &mat,
    const Vec3 &wo, const Vec3 &n, const Vec3 &shading_n, Rng &rng)
{
  BsdfSample out{};

  const bool front_face = dot(wo, n) > 0.f;
  if (!front_face && mat.transmission <= 0.f)
    return out;

  const f32 rough = Kokkos::fmax(mat.roughness, 0.001f);

  f32 w_diffuse, w_specular, w_transmission, w_clearcoat;
  if (front_face) {
    disney_lobe_weights(mat, w_diffuse, w_specular, w_transmission, w_clearcoat);
  } else {
    w_diffuse = 0.f;
    w_specular = 0.f;
    w_transmission = 1.f;
    w_clearcoat = 0.f;
  }

  const f32 sum_w = w_diffuse + w_specular + w_transmission + w_clearcoat;
  if (sum_w <= 0.f)
    return out;

  const f32 p_diffuse = w_diffuse / sum_w;
  const f32 p_specular = w_specular / sum_w;
  const f32 p_transmission = w_transmission / sum_w;
  const f32 p_clearcoat = w_clearcoat / sum_w;

  const f32 xi = rng.next_f32();

  if (xi < p_diffuse) {
    auto s = sample_cosine_hemisphere(shading_n, rng);
    out.wi = s.direction;
    out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
    out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    return out;
  }

  if (xi < p_diffuse + p_specular) {
    if (mat.roughness < 0.001f) {
      out.wi = reflect(-wo, n);
      if (dot(out.wi, n) <= 0.f)
        return BsdfSample{};

      const f32 f0d = dielectric_f0(mat.ior);
      Vec3 F0{f0d, f0d, f0d};
      F0 = lerp(F0, mat.base_color, mat.metallic);

      const Vec3 F = fresnel_schlick(saturate(dot(wo, n)), F0);
      out.f = F;
      out.pdf = 1.f;
      out.is_specular = true;
      return out;
    }

    const Vec3 h = sample_ggx_half_vector(shading_n, rough, rng);
    out.wi = reflect(-wo, h);
    if (dot(out.wi, n) <= 0.f)
      return BsdfSample{};

    out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
    out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    return out;
  }

  if (xi < p_diffuse + p_specular + p_transmission && mat.transmission > 0.f) {
    const Vec3 nn = front_face ? n : n * -1.f;
    const f32 eta = front_face ? (1.f / mat.ior) : mat.ior;

    if (mat.roughness < 0.1f) {
      const Vec3 wt = refract(-wo, nn, eta);
      if (wt.x == 0.f && wt.y == 0.f && wt.z == 0.f) {
        out.wi = reflect(-wo, nn);
      } else {
        out.wi = wt;
      }
      out.pdf = 1.f;
      out.is_specular = true;
      out.f = Vec3{1.f, 1.f, 1.f};
      return out;
    }

    const Vec3 sn_t = front_face ? shading_n : shading_n * -1.f;
    const Vec3 h = sample_ggx_half_vector(sn_t, rough, rng);
    const Vec3 wt = refract(-wo, h, eta);
    if (wt.x == 0.f && wt.y == 0.f && wt.z == 0.f) {
      out.wi = reflect(-wo, h);
      if (dot(out.wi, nn) <= 0.f)
        return BsdfSample{};
      out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
      out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    } else {
      out.wi = wt;
      out.f = mat.base_color;
      const f32 NoH = saturate(Kokkos::fabs(dot(sn_t, h)));
      const f32 VoH = saturate(Kokkos::fabs(dot(wo, h)));
      out.pdf = Kokkos::fmax(D_ggx(NoH, rough) * NoH / (4.f * Kokkos::fmax(VoH, 1e-7f)), 1e-7f);
    }
    return out;
  }

  if (p_clearcoat > 0.f) {
    const f32 a_cc = Kokkos::fmax(mat.clearcoat_roughness, 0.001f);
    const Vec3 h = sample_ggx_half_vector(shading_n, a_cc, rng);
    out.wi = reflect(-wo, h);
    if (dot(out.wi, n) <= 0.f)
      return BsdfSample{};
    out.f = disney_bsdf_eval(mat, wo, out.wi, n, shading_n);
    out.pdf = disney_bsdf_pdf(mat, wo, out.wi, n, shading_n);
    return out;
  }

  return out;
}

}
