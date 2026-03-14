#include "photon/pt/pathtracer.h"

#include <Kokkos_Core.hpp>

#include "photon/pt/disney_bsdf.h"
#include "photon/pt/environment_map.h"
#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/rng.h"
#include "photon/pt/sampling.h"

namespace photon::pt {

void PathTracer::set_scene(const Scene &scene)
{
  m_scene = scene;
}

void PathTracer::set_backend(std::unique_ptr<RayBackend> backend)
{
  m_backend = std::move(backend);
}

static KOKKOS_FUNCTION Vec3 eval_sky_gradient(const Vec3 &dir)
{
  const f32 t = 0.5f * (dir.y + 1.f);
  return (1.f - t) * Vec3{1.f, 1.f, 1.f} + t * Vec3{0.5f, 0.7f, 1.f};
}

RenderResult PathTracer::render() const
{
  const u32 w = params.width;
  const u32 h = params.height;
  const u32 spp = params.samples_per_pixel;
  const u32 max_depth = params.max_depth;

  RenderResult out;
  out.color = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("pt_color", h, w);
  out.depth = Kokkos::View<f32 **, Kokkos::LayoutRight>("pt_depth", h, w);
  out.normal = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("pt_normal", h, w);
  out.albedo = Kokkos::View<Vec3 **, Kokkos::LayoutRight>("pt_albedo", h, w);

  if (!m_backend) {
    Kokkos::deep_copy(out.color, Vec3{0.f, 0.f, 0.f});
    Kokkos::deep_copy(out.depth, 0.f);
    Kokkos::deep_copy(out.normal, Vec3{0.f, 0.f, 0.f});
    Kokkos::deep_copy(out.albedo, Vec3{0.f, 0.f, 0.f});
    return out;
  }

  const Scene scene = m_scene;
  const Camera cam = camera;

  const u32 pixel_count = w * h;

  Kokkos::View<Vec3 *> accum("pt_accum", pixel_count);
  Kokkos::deep_copy(accum, Vec3{0.f, 0.f, 0.f});

  Kokkos::View<Vec3 *> accum_sq("pt_accum_sq", pixel_count);
  Kokkos::deep_copy(accum_sq, Vec3{0.f, 0.f, 0.f});
  Kokkos::View<u32 *> pixel_samples("pt_pixel_spp", pixel_count);
  Kokkos::deep_copy(pixel_samples, 0u);

  Kokkos::View<Vec3 *> aov_albedo("pt_aov_albedo", pixel_count);
  Kokkos::View<Vec3 *> aov_normal("pt_aov_normal", pixel_count);
  Kokkos::View<f32 *> aov_depth("pt_aov_depth", pixel_count);
  Kokkos::View<u32 *> aov_written("pt_aov_written", pixel_count);
  Kokkos::deep_copy(aov_albedo, Vec3{0.f, 0.f, 0.f});
  Kokkos::deep_copy(aov_normal, Vec3{0.f, 0.f, 0.f});
  Kokkos::deep_copy(aov_depth, 0.f);
  Kokkos::deep_copy(aov_written, 0u);

  RayBatch rays;
  rays.origins = Kokkos::View<Vec3 *>("pt_ray_o", pixel_count);
  rays.directions = Kokkos::View<Vec3 *>("pt_ray_d", pixel_count);
  rays.tmin = Kokkos::View<f32 *>("pt_ray_tmin", pixel_count);
  rays.tmax = Kokkos::View<f32 *>("pt_ray_tmax", pixel_count);
  rays.count = pixel_count;

  HitBatch hits;
  hits.hits = Kokkos::View<HitResult *>("pt_hits", pixel_count);
  hits.count = pixel_count;

  Kokkos::View<Vec3 *> throughput("pt_throughput", pixel_count);
  Kokkos::View<u32 *> active("pt_active", pixel_count);

  RayBatch shadow_rays;
  shadow_rays.origins = Kokkos::View<Vec3 *>("pt_shadow_o", pixel_count);
  shadow_rays.directions = Kokkos::View<Vec3 *>("pt_shadow_d", pixel_count);
  shadow_rays.tmin = Kokkos::View<f32 *>("pt_shadow_tmin", pixel_count);
  shadow_rays.tmax = Kokkos::View<f32 *>("pt_shadow_tmax", pixel_count);
  shadow_rays.count = pixel_count;

  Kokkos::View<Vec3 *> nee_contrib("pt_nee", pixel_count);
  Kokkos::View<u32 *> nee_active("pt_nee_active", pixel_count);
  Kokkos::View<u32 *> occluded("pt_occluded", pixel_count);
  Kokkos::View<Vec3 *> sample_start("pt_sample_start", pixel_count);

  const f32 adaptive_threshold = 0.01f;
  const u32 adaptive_min_samples = 16u;

  for (u32 s = 0; s < spp; ++s) {
    Kokkos::parallel_for(
        "pt_init_sample",
        Kokkos::RangePolicy<>(0, int(pixel_count)),
        KOKKOS_LAMBDA(const int idx) {
          const u32 x = u32(idx) % w;
          const u32 y = u32(idx) / w;

          if (s >= adaptive_min_samples) {
            const u32 ns = pixel_samples(idx);
            if (ns >= adaptive_min_samples) {
              const Vec3 mean = accum(idx) * (1.f / f32(ns));
              const Vec3 sq = accum_sq(idx) * (1.f / f32(ns));
              const Vec3 var = sq - Vec3{mean.x * mean.x, mean.y * mean.y, mean.z * mean.z};
              const f32 max_var = Kokkos::fmax(var.x, Kokkos::fmax(var.y, var.z));
              const f32 lum = Kokkos::fmax(0.2126f * mean.x + 0.7152f * mean.y + 0.0722f * mean.z, 1e-4f);
              if (max_var / (lum * lum) < adaptive_threshold) {
                active(idx) = 0u;
                rays.tmax(idx) = 0.f;
                return;
              }
            }
          }

          Rng rng(u32(1337u) ^ (u32(x) * 9781u) ^ (u32(y) * 6271u) ^ (s * 26699u));
          const f32 u = (f32(x) + rng.next_f32()) / f32(w - 1);
          const f32 v = (f32(y) + rng.next_f32()) / f32(h - 1);

          const Ray r = cam.ray(u, 1.f - v, rng);
          rays.origins(idx) = r.org;
          rays.directions(idx) = r.dir;
          rays.tmin(idx) = 1e-3f;
          rays.tmax(idx) = 1e30f;

          throughput(idx) = Vec3{1.f, 1.f, 1.f};
          active(idx) = 1u;
          sample_start(idx) = accum(idx);
        });

    for (u32 bounce = 0; bounce < max_depth; ++bounce) {
      m_backend->trace_closest(rays, hits);

      Kokkos::deep_copy(nee_contrib, Vec3{0.f, 0.f, 0.f});
      Kokkos::deep_copy(nee_active, 0u);

      const auto materials = scene.materials;
      const auto lights = scene.lights;
      const auto scene_textures = scene.textures;
      const u32 light_count = scene.light_count;
      const bool has_env_map = scene.env_map.has_value();
      const EnvironmentMap env_map_val = has_env_map ? scene.env_map.value() : EnvironmentMap{};

      Kokkos::parallel_for(
          "pt_shade",
          Kokkos::RangePolicy<>(0, int(pixel_count)),
          KOKKOS_LAMBDA(const int idx) {
            if (active(idx) == 0u)
              return;

            HitResult hr = hits.hits(idx);

            if (!hr.hit) {
              const Vec3 dir = rays.directions(idx);
              Vec3 L{0.01f, 0.01f, 0.01f};

              if (has_env_map) {
                L = env_map_val.evaluate(dir);
              } else if (light_count == 0) {
                L = eval_sky_gradient(dir);
              }

              accum(idx) = accum(idx) + throughput(idx) * L;
              active(idx) = 0u;
              return;
            }

            const Vec3 p = hr.position;
            const Vec3 wo = normalize(-rays.directions(idx));

            Vec3 n = hr.normal;
            Vec3 sn = hr.shading_normal;

            Material mat_pre = materials.extent(0) ? materials(hr.material_id) : Material{};

            f32 alpha_val = mat_pre.alpha;
            if (mat_pre.alpha_tex >= 0 && scene_textures.count > 0) {
              alpha_val = scene_textures.sample(mat_pre.alpha_tex, hr.uv.x, hr.uv.y).x;
            }

            if (alpha_val < 1.f) {
              Rng alpha_rng(u32(1337u) ^ (u32(idx) * 7919u) ^ (bounce * 6271u) ^ (s * 26699u));
              if (alpha_rng.next_f32() > alpha_val) {
                rays.origins(idx) = p;
                rays.tmin(idx) = 1e-3f;
                rays.tmax(idx) = 1e30f;
                return;
              }
            }

            const bool is_transmissive = mat_pre.transmission > 0.f;

            if (!is_transmissive && dot(n, wo) < 0.f) {
              n = n * -1.f;
              sn = sn * -1.f;
            }

            Material mat = materials.extent(0) ? materials(hr.material_id) : Material{};

            if (mat.base_color_tex >= 0 && scene_textures.count > 0) {
              mat.base_color = scene_textures.sample(mat.base_color_tex, hr.uv.x, hr.uv.y);
            } else if (hr.has_interpolated_color) {
              mat.base_color = hr.interpolated_color;
            }

            if (mat.normal_tex >= 0 && scene_textures.count > 0) {
              Vec3 nm = scene_textures.sample(mat.normal_tex, hr.uv.x, hr.uv.y);
              nm = nm * 2.f - Vec3{1.f, 1.f, 1.f};
              Vec3 tangent{}, bitangent{};
              onb_from_normal(sn, tangent, bitangent);
              sn = normalize(tangent * nm.x + bitangent * nm.y + sn * nm.z);
            }

            if (mat.roughness_tex >= 0 && scene_textures.count > 0) {
              Vec3 rt = scene_textures.sample(mat.roughness_tex, hr.uv.x, hr.uv.y);
              mat.roughness = rt.x;
            }

            if (mat.metallic_tex >= 0 && scene_textures.count > 0) {
              Vec3 mt = scene_textures.sample(mat.metallic_tex, hr.uv.x, hr.uv.y);
              mat.metallic = mt.x;
            }

            if (bounce == 0 && aov_written(idx) == 0u) {
              aov_written(idx) = 1u;
              aov_depth(idx) = hr.t;
              aov_normal(idx) = normalize(sn);
              aov_albedo(idx) = mat.base_color;
            }

            if (material_is_emissive(mat)) {
              accum(idx) = accum(idx) + throughput(idx) * material_emission(mat);
            }

            Rng rng(u32(1337u) ^ (u32(idx) * 9781u) ^ (bounce * 6271u) ^ (s * 26699u));

            if (!material_is_emissive(mat)) {
              if (light_count > 0 && mat.roughness >= 0.001f) {
                const u32 light_idx = u32(rng.next_f32() * f32(light_count));
                const Light light = lights(light_idx);
                const LightSample ls = sample_light(light, p, rng);

                // Two-sided NEE: flip normals toward the light if needed
                const f32 nee_cos = dot(ls.wi, n);
                const Vec3 n_l = nee_cos >= 0.f ? n : n * -1.f;
                const Vec3 sn_l = nee_cos >= 0.f ? sn : sn * -1.f;
                const Vec3 wo_l = nee_cos >= 0.f ? wo : wo * -1.f;

                if (ls.pdf > 0.f && (ls.Li.x > 0.f || ls.Li.y > 0.f || ls.Li.z > 0.f) && Kokkos::fabs(nee_cos) > 0.f) {
                  const Vec3 f = disney_bsdf_eval(mat, wo_l, ls.wi, n_l, sn_l);
                  const f32 bsdf_pdf = disney_bsdf_pdf(mat, wo_l, ls.wi, n_l, sn_l);
                  const f32 w_mis = ls.is_delta ? 1.f : power_heuristic(1, ls.pdf, 1, bsdf_pdf);
                  const f32 inv_light_pdf = 1.f / (ls.pdf * f32(light_count));
                  const Vec3 nee = throughput(idx) * f * ls.Li * (w_mis * inv_light_pdf);

                  nee_contrib(idx) = nee;
                  nee_active(idx) = 1u;

                  shadow_rays.origins(idx) = p;
                  shadow_rays.directions(idx) = ls.wi;
                  shadow_rays.tmin(idx) = 1e-3f;
                  shadow_rays.tmax(idx) = ls.dist * 0.99f;
                }
              }

              if (has_env_map && mat.roughness >= 0.001f) {
                f32 env_pdf = 0.f;
                const Vec3 wi = env_map_val.sample_direction(rng, env_pdf);
                if (env_pdf > 0.f && dot(wi, n) > 0.f) {
                  const Vec3 Li = env_map_val.evaluate(wi);
                  const Vec3 f = disney_bsdf_eval(mat, wo, wi, n, sn);
                  const f32 bsdf_pdf = disney_bsdf_pdf(mat, wo, wi, n, sn);
                  const f32 w_mis = power_heuristic(1, env_pdf, 1, bsdf_pdf);
                  const Vec3 nee = throughput(idx) * f * Li * (w_mis / env_pdf);

                  nee_contrib(idx) = nee_contrib(idx) + nee;
                  nee_active(idx) = 1u;

                  shadow_rays.origins(idx) = p;
                  shadow_rays.directions(idx) = wi;
                  shadow_rays.tmin(idx) = 1e-3f;
                  shadow_rays.tmax(idx) = 1e30f;
                }
              }
            }

            const BsdfSample bs = disney_bsdf_sample(mat, wo, n, sn, rng);
            if (bs.pdf <= 0.f || (bs.f.x == 0.f && bs.f.y == 0.f && bs.f.z == 0.f)) {
              active(idx) = 0u;
              return;
            }

            if (bounce >= 2) {
              const Vec3 tp = throughput(idx);
              const f32 max_tp = Kokkos::fmax(tp.x, Kokkos::fmax(tp.y, tp.z));
              const f32 q = Kokkos::fmin(max_tp, 0.95f);
              if (rng.next_f32() > q) {
                active(idx) = 0u;
                return;
              }
              throughput(idx) = tp * (1.f / q);
            }

            throughput(idx) = throughput(idx) * (bs.f * (1.f / bs.pdf));

            {
              const Vec3 tp = throughput(idx);
              const f32 max_clamp = 10.f;
              if (tp.x > max_clamp || tp.y > max_clamp || tp.z > max_clamp) {
                const f32 m = Kokkos::fmax(tp.x, Kokkos::fmax(tp.y, tp.z));
                throughput(idx) = tp * (max_clamp / m);
              }
            }

            const Vec3 wi = normalize(bs.wi);
            const bool entering = dot(wi, hr.normal) < 0.f;
            const Vec3 offset = entering ? hr.normal * -1e-3f : hr.normal * 1e-3f;
            rays.origins(idx) = bs.is_specular && is_transmissive ? p + offset : p;
            rays.directions(idx) = wi;
            rays.tmin(idx) = 1e-3f;
            rays.tmax(idx) = 1e30f;
          });

      m_backend->trace_occluded(shadow_rays, occluded);

      Kokkos::parallel_for(
          "pt_nee_accum",
          Kokkos::RangePolicy<>(0, int(pixel_count)),
          KOKKOS_LAMBDA(const int idx) {
            if (active(idx) == 0u)
              return;
            if (nee_active(idx) == 0u)
              return;
            if (occluded(idx) == 0u) {
              accum(idx) = accum(idx) + nee_contrib(idx);
            }
          });

      Kokkos::parallel_for(
          "pt_pack_active",
          Kokkos::RangePolicy<>(0, int(pixel_count)),
          KOKKOS_LAMBDA(const int idx) {
            if (active(idx) == 0u) {
              rays.tmax(idx) = 0.f;
              return;
            }
          });
    }

    Kokkos::parallel_for(
        "pt_update_variance",
        Kokkos::RangePolicy<>(0, int(pixel_count)),
        KOKKOS_LAMBDA(const int idx) {
          const Vec3 sample_val = accum(idx) - sample_start(idx);
          accum_sq(idx) = accum_sq(idx) + Vec3{
            sample_val.x * sample_val.x,
            sample_val.y * sample_val.y,
            sample_val.z * sample_val.z};
          pixel_samples(idx) = pixel_samples(idx) + 1u;
        });
  }

  Kokkos::parallel_for(
      "pt_resolve",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(h), int(w)}),
      KOKKOS_LAMBDA(const int y, const int x) {
        const int idx = y * int(w) + x;
        const u32 ns = pixel_samples(idx);
        const f32 inv = ns > 0 ? (1.f / f32(ns)) : 0.f;
        out.color(y, x) = accum(idx) * inv;
        out.depth(y, x) = aov_depth(idx);
        out.normal(y, x) = aov_normal(idx);
        out.albedo(y, x) = aov_albedo(idx);
      });

  Kokkos::fence();
  return out;
}

} // namespace photon::pt
