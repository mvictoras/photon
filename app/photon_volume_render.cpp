#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "photon/pt/camera.h"
#include "photon/pt/math.h"
#include "photon/pt/ray.h"
#include "photon/pt/rng.h"
#include "photon/pt/volume.h"

namespace {

using namespace photon::pt;

static float aces_filmic(float x)
{
  const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
  x = std::fmax(x, 0.f);
  return std::fmin((x * (a * x + b)) / (x * (c * x + d) + e), 1.f);
}

float noise3d_smooth(float x, float y, float z)
{
  float n = std::sin(x * 7.13f + y * 17.71f + z * 31.37f) * 43758.5453f;
  return (n - std::floor(n)) * 2.f - 1.f;
}

float fbm(float x, float y, float z, int octaves)
{
  float v = 0.f, amp = 1.f, f = 1.f, total_amp = 0.f;
  for (int i = 0; i < octaves; ++i) {
    v += amp * noise3d_smooth(x * f, y * f, z * f);
    total_amp += amp;
    amp *= 0.5f;
    f *= 2.0f;
  }
  return v / total_amp;
}

enum class VolumeType { Cloud, Nebula, Explosion };

void generate_volume(Kokkos::View<f32 ***, Kokkos::LayoutRight> &data,
                     int res, VolumeType type, float &max_density)
{
  auto host = Kokkos::create_mirror_view(data);
  max_density = 0.f;

  for (int z = 0; z < res; ++z)
    for (int y = 0; y < res; ++y)
      for (int x = 0; x < res; ++x) {
        float fx = (float(x) / float(res - 1)) * 2.f - 1.f;
        float fy = (float(y) / float(res - 1)) * 2.f - 1.f;
        float fz = (float(z) / float(res - 1)) * 2.f - 1.f;
        float r = std::sqrt(fx*fx + fy*fy + fz*fz);

        float density = 0.f;

        if (type == VolumeType::Cloud) {
          float sphere = std::fmax(0.f, 1.f - r * 1.2f);
          float turb = fbm(fx * 3.f, fy * 3.f, fz * 3.f, 5) * 0.5f + 0.5f;
          density = sphere * turb;
        } else if (type == VolumeType::Nebula) {
          float shell = std::exp(-4.f * (r - 0.6f) * (r - 0.6f));
          float phi = std::atan2(fz, fx);
          float spiral = std::sin(phi * 2.f + r * 8.f) * 0.5f + 0.5f;
          float turb = fbm(fx * 4.f, fy * 4.f, fz * 4.f, 4) * 0.5f + 0.5f;
          density = shell * (spiral * 0.6f + turb * 0.4f);
          float jet = std::exp(-10.f * (fx*fx + fz*fz)) * std::fmax(0.f, fy) * 2.f;
          density = std::fmax(density, jet * 0.3f);
        } else {
          float core = std::exp(-8.f * r * r);
          float shell = std::exp(-2.f * (r - 0.7f) * (r - 0.7f));
          float turb = fbm(fx * 5.f + 0.5f, fy * 5.f, fz * 5.f + 0.3f, 5) * 0.5f + 0.5f;
          density = (core * 2.f + shell) * turb;
          float tendrils = std::fmax(0.f, fbm(fx*3.f, fy*3.f, fz*3.f, 3));
          density += tendrils * std::fmax(0.f, 1.f - r) * 0.5f;
        }

        density = std::fmax(0.f, density);
        host(z, y, x) = density;
        max_density = std::fmax(max_density, density);
      }

  Kokkos::deep_copy(data, host);
}

}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    int res = 128;
    int width = 1920, height = 1080;
    int spp = 64;
    float exposure = 1.0f;
    std::string output = "volume_render.ppm";
    std::string dataset = "cloud";
    VolumeType vol_type = VolumeType::Cloud;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--res" && i+1 < argc) res = std::atoi(argv[++i]);
      else if (arg == "--width" && i+1 < argc) width = std::atoi(argv[++i]);
      else if (arg == "--height" && i+1 < argc) height = std::atoi(argv[++i]);
      else if (arg == "--spp" && i+1 < argc) spp = std::atoi(argv[++i]);
      else if (arg == "--exposure" && i+1 < argc) exposure = std::atof(argv[++i]);
      else if (arg == "-o" && i+1 < argc) output = argv[++i];
      else if (arg == "--dataset" && i+1 < argc) {
        dataset = argv[++i];
        if (dataset == "cloud") vol_type = VolumeType::Cloud;
        else if (dataset == "nebula") vol_type = VolumeType::Nebula;
        else if (dataset == "explosion") vol_type = VolumeType::Explosion;
      }
    }

    std::fprintf(stderr, "Volume render: %s, %d³, %dx%d @ %d spp\n",
        dataset.c_str(), res, width, height, spp);

    VolumeGrid vol;
    vol.bounds_lo = {-1.f, -1.f, -1.f};
    vol.bounds_hi = {1.f, 1.f, 1.f};
    vol.density = Kokkos::View<f32 ***, Kokkos::LayoutRight>("density", res, res, res);

    generate_volume(vol.density, res, vol_type, vol.max_density);

    if (vol_type == VolumeType::Cloud) {
      vol.sigma_s = {2.f, 2.f, 2.f};
      vol.sigma_a = {0.005f, 0.005f, 0.005f};
      vol.g = 0.76f;
    } else if (vol_type == VolumeType::Nebula) {
      vol.sigma_s = {0.8f, 0.5f, 1.2f};
      vol.sigma_a = {0.1f, 0.02f, 0.01f};
      vol.g = 0.4f;
      vol.emission = {0.4f, 0.1f, 0.8f};
      vol.emission_strength = 0.5f;
    } else {
      vol.sigma_s = {3.f, 1.5f, 0.5f};
      vol.sigma_a = {0.2f, 0.1f, 0.02f};
      vol.g = -0.3f;
      vol.emission = {3.f, 1.2f, 0.3f};
      vol.emission_strength = 1.f;
    }

    std::fprintf(stderr, "Generated %s volume: max_density=%.2f\n", dataset.c_str(), vol.max_density);

    float aspect = float(width) / float(height);
    Vec3 cam_pos = {0.f, 0.f, 3.5f};
    Vec3 cam_target = {0.f, 0.f, 0.f};
    Camera cam = Camera::make_perspective(cam_pos, cam_target, {0,1,0}, 45.f, aspect);

    Kokkos::View<Vec3 **, Kokkos::LayoutRight> framebuffer("fb", height, width);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < spp; ++s) {
      Kokkos::parallel_for("volume_render",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
          KOKKOS_LAMBDA(const int y, const int x) {
            Rng rng(u32(42u) ^ (u32(x) * 9781u) ^ (u32(y) * 6271u) ^ (u32(s) * 26699u));
            f32 u = (f32(x) + rng.next_f32()) / f32(width - 1);
            f32 v = (f32(y) + rng.next_f32()) / f32(height - 1);
            Ray ray = cam.ray(u, 1.f - v, rng);

            f32 t_enter, t_exit;
            if (!vol.intersect_bounds(ray, t_enter, t_exit)) {
              return;
            }
            if (t_enter < 1e-3f) t_enter = 1e-3f;

            Vec3 throughput{1.f, 1.f, 1.f};
            Vec3 L{0.f, 0.f, 0.f};

            Vec3 dir = ray.dir;
            Vec3 org = ray.org;
            f32 tmin = t_enter, tmax = t_exit;

            for (int bounce = 0; bounce < 32; ++bounce) {
              Ray march_ray{org, dir};
              VolumeEvent ve = delta_tracking(vol, march_ray, tmin, tmax, rng);

              if (!ve.scattered) {
                Vec3 sky_dir = normalize(dir);
                f32 sky_t = 0.5f * (sky_dir.y + 1.f);
                Vec3 sky = Vec3{0.5f, 0.7f, 1.0f} * sky_t + Vec3{1.f, 1.f, 1.f} * (1.f - sky_t);
                L = L + throughput * sky * 0.5f;
                break;
              }

              Vec3 sigma_t_v = ve.sigma_s + ve.sigma_a;
              f32 scatter_albedo = max_component(ve.sigma_s) /
                  (max_component(sigma_t_v) + 1e-8f);

              if (vol.emission_strength > 0.f) {
                f32 d = vol.sample_density(ve.position);
                L = L + throughput * vol.emission * (vol.emission_strength * d);
              }

              Vec3 sun_dir = normalize(Vec3{0.5f, 1.f, 0.3f});
              f32 sun_t_enter, sun_t_exit;
              Ray sun_ray{ve.position, sun_dir};
              if (vol.intersect_bounds(sun_ray, sun_t_enter, sun_t_exit)) {
                f32 tau = 0.f;
                f32 sun_step = (sun_t_exit - Kokkos::fmax(sun_t_enter, 0.f)) / 16.f;
                for (int si = 0; si < 16; ++si) {
                  f32 st = Kokkos::fmax(sun_t_enter, 0.f) + (f32(si) + 0.5f) * sun_step;
                  Vec3 sp = ve.position + sun_dir * st;
                  tau += vol.sample_density(sp) * max_component(sigma_t_v) * sun_step;
                }
                f32 sun_transmittance = Kokkos::exp(-tau);
                f32 cos_theta = dot(normalize(dir * -1.f), sun_dir);
                f32 phase = pdf_hg(cos_theta, vol.g);
                Vec3 sun_color = {10.f, 9.f, 7.f};
                L = L + throughput * sun_color * (sun_transmittance * phase * scatter_albedo);
              }

              throughput = throughput * scatter_albedo;
              if (max_component(throughput) < 0.01f) break;

              Vec3 wo = normalize(dir * -1.f);
              dir = sample_hg(wo, vol.g, rng);
              org = ve.position;
              tmin = 1e-3f;

              if (!vol.intersect_bounds(Ray{org, dir}, t_enter, t_exit)) break;
              tmax = t_exit;
            }

            framebuffer(y, x) = framebuffer(y, x) + L;
          });
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "Rendered in %.1f ms\n", ms);

    auto fb_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, framebuffer);
    float inv_spp = 1.f / float(spp);

    std::ofstream out(output, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    std::vector<unsigned char> row(size_t(width) * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Vec3 c = fb_h(y, x) * inv_spp;
        auto to_u8 = [exposure](float v) -> unsigned char {
          v = aces_filmic(v * exposure);
          v = std::pow(v, 1.f / 2.2f);
          return static_cast<unsigned char>(v * 255.f + 0.5f);
        };
        row[3*x]   = to_u8(c.x);
        row[3*x+1] = to_u8(c.y);
        row[3*x+2] = to_u8(c.z);
      }
      out.write(reinterpret_cast<char*>(row.data()), std::streamsize(row.size()));
    }
    std::fprintf(stderr, "Output: %s\n", output.c_str());
  }
  Kokkos::finalize();
  return 0;
}
