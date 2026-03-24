#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "photon/pbrt/pbrt_parser.h"
#include "photon/pbrt/pbrt_to_photon.h"
#include "photon/pt/backend/ray_backend.h"
#include "photon/pt/bdpt.h"
#include "photon/pt/camera.h"
#include "photon/pt/denoiser.h"
#include "photon/pt/disney_bsdf.h"
#include "photon/pt/light.h"
#include "photon/pt/math.h"
#include "photon/pt/pathtracer.h"
#include "photon/pt/rng.h"
#include "photon/pt/scene.h"

namespace {

using namespace photon::pt;

static float aces_filmic(float x)
{
  const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  x = std::fmax(x, 0.f);
  return std::fmin((x * (a * x + b)) / (x * (c * x + d) + e), 1.f);
}

struct Args {
  std::string scene_path;
  std::string output = "bdpt_render.ppm";
  int width = -1, height = -1, spp = -1, max_depth = -1;
  float exposure = 1.f;
  bool denoise = false;
};

Args parse_args(int argc, char **argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i+1 < argc) a.output = argv[++i];
    else if (arg == "--width" && i+1 < argc) a.width = std::atoi(argv[++i]);
    else if (arg == "--height" && i+1 < argc) a.height = std::atoi(argv[++i]);
    else if (arg == "--spp" && i+1 < argc) a.spp = std::atoi(argv[++i]);
    else if (arg == "--max-depth" && i+1 < argc) a.max_depth = std::atoi(argv[++i]);
    else if (arg == "--exposure" && i+1 < argc) a.exposure = std::atof(argv[++i]);
    else if (arg == "--denoise") a.denoise = true;
    else if (arg[0] != '-') a.scene_path = arg;
  }
  return a;
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    Args args = parse_args(argc, argv);
    if (args.scene_path.empty()) {
      std::fprintf(stderr, "Usage: photon_bdpt_render <scene.pbrt> [options]\n");
      Kokkos::finalize();
      return 1;
    }

    std::fprintf(stderr, "BDPT Render: %s\n", args.scene_path.c_str());

    auto pbrt_scene = photon::pbrt::parse_pbrt_file(args.scene_path);
    if (args.width > 0) pbrt_scene.width = args.width;
    if (args.height > 0) pbrt_scene.height = args.height;
    if (args.spp > 0) pbrt_scene.spp = args.spp;
    if (args.max_depth > 0) pbrt_scene.max_depth = args.max_depth;

    std::string base_dir;
    auto slash = args.scene_path.find_last_of("/\\");
    if (slash != std::string::npos) base_dir = args.scene_path.substr(0, slash + 1);

    auto converted = photon::pbrt::convert_pbrt_scene(pbrt_scene, base_dir);
    const Scene &scene = converted.scene;
    const Camera &cam = converted.camera;
    const u32 W = u32(pbrt_scene.width), H = u32(pbrt_scene.height);
    const u32 spp = u32(pbrt_scene.spp);
    const u32 max_depth = u32(pbrt_scene.max_depth);

    std::fprintf(stderr, "Scene: %u tris, %u mats, %u lights, %dx%d @ %d spp\n",
        scene.mesh.triangle_count(), scene.material_count, scene.light_count, W, H, spp);

    if (scene.light_count == 0) {
      std::fprintf(stderr, "Warning: no lights in scene, BDPT will not produce light paths\n");
    }

    auto backend = create_best_backend();
    backend->build_accel(scene);

    Kokkos::View<Vec3 **, Kokkos::LayoutRight> color_accum("bdpt_color", H, W);
    Kokkos::View<u32 *, Kokkos::LayoutRight> splat_count("splat_count", H * W);
    Kokkos::View<Vec3 *> splat_accum("splat_accum", H * W);
    Kokkos::deep_copy(color_accum, Vec3{0.f, 0.f, 0.f});
    Kokkos::deep_copy(splat_accum, Vec3{0.f, 0.f, 0.f});
    Kokkos::deep_copy(splat_count, 0u);

    const u32 pixel_count = W * H;

    RayBatch rays;
    rays.origins = Kokkos::View<Vec3 *>("o", pixel_count);
    rays.directions = Kokkos::View<Vec3 *>("d", pixel_count);
    rays.tmin = Kokkos::View<f32 *>("tmin", pixel_count);
    rays.tmax = Kokkos::View<f32 *>("tmax", pixel_count);
    rays.count = pixel_count;

    HitBatch hits;
    hits.hits = Kokkos::View<HitResult *>("h", pixel_count);
    hits.count = pixel_count;

    Kokkos::View<Vec3 *> shadow_origins("so", pixel_count);
    Kokkos::View<Vec3 *> shadow_dirs("sd", pixel_count);
    Kokkos::View<f32 *> shadow_tmin("stmin", pixel_count);
    Kokkos::View<f32 *> shadow_tmax("stmax", pixel_count);
    Kokkos::View<u32 *> shadow_active("sa", pixel_count);
    Kokkos::View<u32 *> occluded("occ", pixel_count);

    RayBatch shadow_rays;
    shadow_rays.origins = shadow_origins;
    shadow_rays.directions = shadow_dirs;
    shadow_rays.tmin = shadow_tmin;
    shadow_rays.tmax = shadow_tmax;
    shadow_rays.count = pixel_count;

    auto t0 = std::chrono::high_resolution_clock::now();

    const auto lights = scene.lights;
    const auto materials = scene.materials;
    const auto scene_textures = scene.textures;
    const u32 light_count = scene.light_count;
    const bool has_env_map = scene.env_map.has_value();
    const EnvironmentMap env_map_val = has_env_map ? scene.env_map.value() : EnvironmentMap{};
    const auto mesh_positions = scene.mesh.positions;
    const auto mesh_indices = scene.mesh.indices;
    const auto mesh_mat_ids = scene.mesh.material_ids;
    const auto emissive_ids = scene.emissive_prim_ids;
    const u32 emissive_count = scene.emissive_count;

    for (u32 s = 0; s < spp; ++s) {

      // ===== PT camera path (standard path tracing contribution) =====
      Kokkos::parallel_for("bdpt_init", pixel_count, KOKKOS_LAMBDA(const int idx) {
        const u32 x = u32(idx) % W, y = u32(idx) / W;
        Rng rng(u32(1337u) ^ (x * 9781u) ^ (y * 6271u) ^ (s * 26699u));
        const f32 u = (f32(x) + rng.next_f32()) / f32(W - 1);
        const f32 v = (f32(y) + rng.next_f32()) / f32(H - 1);
        Ray r = cam.ray(u, 1.f - v, rng);
        rays.origins(idx) = r.org;
        rays.directions(idx) = r.dir;
        rays.tmin(idx) = 1e-3f;
        rays.tmax(idx) = 1e30f;
        shadow_active(idx) = 0u;
      });

      Kokkos::View<Vec3 *> throughput("tp", pixel_count);
      Kokkos::View<u32 *> active("act", pixel_count);
      Kokkos::deep_copy(throughput, Vec3{1.f, 1.f, 1.f});
      Kokkos::deep_copy(active, 1u);

      for (u32 bounce = 0; bounce < max_depth; ++bounce) {
        backend->trace_closest(rays, hits);

        Kokkos::View<Vec3 *> nee_contrib("nee", pixel_count);
        Kokkos::View<u32 *> nee_active("na", pixel_count);
        Kokkos::deep_copy(nee_contrib, Vec3{0.f, 0.f, 0.f});
        Kokkos::deep_copy(nee_active, 0u);

        Kokkos::parallel_for("bdpt_shade", pixel_count, KOKKOS_LAMBDA(const int idx) {
          if (active(idx) == 0u) return;

          HitResult hr = hits.hits(idx);
          const Vec3 wo = normalize(rays.directions(idx) * -1.f);

          if (!hr.hit) {
            if (has_env_map) {
              color_accum(u32(idx)/W, u32(idx)%W) = color_accum(u32(idx)/W, u32(idx)%W)
                  + throughput(idx) * env_map_val.evaluate(rays.directions(idx));
            }
            active(idx) = 0u;
            return;
          }

          Material mat = materials.extent(0) ? materials(hr.material_id) : Material{};
          if (mat.base_color_tex >= 0 && scene_textures.count > 0)
            mat.base_color = scene_textures.sample(mat.base_color_tex, hr.uv.x, hr.uv.y);
          else if (hr.has_interpolated_color)
            mat.base_color = hr.interpolated_color;

          if (material_is_emissive(mat) && dot(wo, hr.normal) > 0.f) {
            color_accum(u32(idx)/W, u32(idx)%W) = color_accum(u32(idx)/W, u32(idx)%W)
                + throughput(idx) * material_emission(mat);
          }

          Vec3 n = hr.normal, sn = hr.shading_normal;
          const bool is_transmissive = mat.transmission > 0.f;
          if (!is_transmissive && dot(n, wo) < 0.f) { n = n*-1.f; sn = sn*-1.f; }

          if (!material_is_emissive(mat) && light_count > 0 && mat.roughness >= 0.001f) {
            const u32 li = u32(0);
            Rng rng2(u32(8765u) ^ u32(idx) ^ (bounce * 1337u) ^ (s * 26699u));
            LightSample ls = sample_light(lights(li), hr.position, rng2);
            if (ls.pdf > 0.f && (ls.Li.x > 0.f || ls.Li.y > 0.f || ls.Li.z > 0.f)) {
              const Vec3 f = disney_bsdf_eval(mat, wo, ls.wi, n, sn);
              const f32 bsdf_pdf = disney_bsdf_pdf(mat, wo, ls.wi, n, sn);
              const f32 w_mis = power_heuristic(1, ls.pdf, 1, bsdf_pdf);
              nee_contrib(idx) = throughput(idx) * f * ls.Li * (w_mis / (ls.pdf * f32(light_count)));
              nee_active(idx) = 1u;
              shadow_origins(idx) = hr.position;
              shadow_dirs(idx) = ls.wi;
              shadow_tmin(idx) = 1e-3f;
              shadow_tmax(idx) = ls.dist * 0.99f;
            }
          }

          Rng rng3(u32(3141u) ^ u32(idx) ^ (bounce * 9781u) ^ (s * 6271u));
          BsdfSample bs = disney_bsdf_sample(mat, wo, n, sn, rng3);
          if (bs.pdf <= 0.f) { active(idx) = 0u; return; }

          throughput(idx) = throughput(idx) * (bs.f * (1.f / bs.pdf));
          const f32 max_tp = Kokkos::fmax(throughput(idx).x, Kokkos::fmax(throughput(idx).y, throughput(idx).z));
          if (max_tp > 10.f) throughput(idx) = throughput(idx) * (10.f / max_tp);
          if (bounce >= 2 && max_tp < 0.05f) { active(idx) = 0u; return; }

          if (is_transmissive && bs.is_specular) {
            const f32 sign = dot(normalize(bs.wi), hr.normal) < 0.f ? -1.f : 1.f;
            rays.origins(idx) = hr.position + hr.normal * (sign * 1e-3f);
          } else {
            rays.origins(idx) = hr.position;
          }
          rays.directions(idx) = normalize(bs.wi);
          rays.tmin(idx) = 1e-3f;
          rays.tmax(idx) = 1e30f;
        });

        Kokkos::deep_copy(occluded, 0u);
        backend->trace_occluded(shadow_rays, occluded);

        Kokkos::parallel_for("bdpt_nee", pixel_count, KOKKOS_LAMBDA(const int idx) {
          if (nee_active(idx) && occluded(idx) == 0u) {
            color_accum(u32(idx)/W, u32(idx)%W) = color_accum(u32(idx)/W, u32(idx)%W)
                + nee_contrib(idx);
          }
        });
      }

      // ===== BDPT: Light path with s=1 (direct light-to-camera connection) =====
      if (light_count > 0) {
        Kokkos::parallel_for("bdpt_light_path", pixel_count, KOKKOS_LAMBDA(const int idx) {
          Rng rng(u32(9999u) ^ u32(idx) ^ (s * 31337u));

          const Light &lt = lights(0u);
          if (lt.area <= 0.f) return;

          const f32 su = Kokkos::sqrt(rng.next_f32());
          const f32 r2 = rng.next_f32();
          Vec3 light_p = lt.position + lt.edge1 * (1.f - su) + lt.edge2 * (su * r2);
          Vec3 light_n = normalize(cross(lt.edge1, lt.edge2));
          const f32 area_pdf = 1.f / lt.area;
          Vec3 Le = lt.color * lt.intensity;

          f32 rx, ry;
          if (!cam.world_to_raster(light_p, int(W), int(H), rx, ry)) return;

          const int px = int(rx), py = int(ry);
          if (px < 0 || px >= int(W) || py < 0 || py >= int(H)) return;

          Vec3 to_cam = cam.origin - light_p;
          const f32 dist2 = dot(to_cam, to_cam);
          if (dist2 < 1e-8f) return;
          const f32 dist = Kokkos::sqrt(dist2);
          to_cam = to_cam * (1.f / dist);

          const f32 cos_lt = dot(to_cam, light_n);
          if (cos_lt <= 0.f) return;

          const f32 cam_imp = cam.importance(to_cam * -1.f);
          if (cam_imp <= 0.f) return;

          const Vec3 contrib = Le * (cam_imp * cos_lt / (area_pdf * f32(light_count) * dist2));

          Kokkos::atomic_add(&splat_accum(py * W + px).x, contrib.x);
          Kokkos::atomic_add(&splat_accum(py * W + px).y, contrib.y);
          Kokkos::atomic_add(&splat_accum(py * W + px).z, contrib.z);
        });
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "Rendered in %.1f ms\n", ms);

    Kokkos::View<Vec3 **, Kokkos::LayoutRight, Kokkos::HostSpace> color_h("ch", H, W);
    auto splat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, splat_accum);
    {
      auto ca_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, color_accum);
      for (u32 y = 0; y < H; ++y)
        for (u32 x = 0; x < W; ++x)
          color_h(y, x) = ca_h(y, x) * (1.f / f32(spp))
                        + splat_h(y * W + x) * (1.f / f32(spp));
    }

    std::ofstream out(args.output, std::ios::binary);
    out << "P6\n" << W << " " << H << "\n255\n";
    std::vector<unsigned char> row(size_t(W) * 3);
    for (u32 y = 0; y < H; ++y) {
      for (u32 x = 0; x < W; ++x) {
        Vec3 c = color_h(y, x);
        auto to_u8 = [&](float v) -> unsigned char {
          v = aces_filmic(v * args.exposure);
          v = std::pow(v, 1.f / 2.2f);
          return (unsigned char)(v * 255.f + 0.5f);
        };
        row[3*x] = to_u8(c.x); row[3*x+1] = to_u8(c.y); row[3*x+2] = to_u8(c.z);
      }
      out.write((char*)row.data(), W * 3);
    }
    std::fprintf(stderr, "Output: %s\n", args.output.c_str());
  }
  Kokkos::finalize();
  return 0;
}
