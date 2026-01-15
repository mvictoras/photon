#include "opencode/pt/pathtracer.h"

namespace opencode::pt {

KOKKOS_FUNCTION Vec3 PathTracer::shade(Rng &rng, Ray ray) const
{
  Sphere spheres[2];
  spheres[0] = Sphere{{0.f, 0.f, -1.f}, 0.5f, {0.7f, 0.3f, 0.3f}};
  spheres[1] = Sphere{{0.f, -100.5f, -1.f}, 100.f, {0.8f, 0.8f, 0.0f}};

  Vec3 throughput{1.f, 1.f, 1.f};
  Vec3 radiance{0.f, 0.f, 0.f};

  for (u32 depth = 0; depth < params.max_depth; ++depth) {
    Hit best;
    best.t = 1e30f;

    for (const auto &s : spheres) {
      const Hit h = s.intersect(ray, 1e-3f, best.t);
      if (h.hit)
        best = h;
    }

    if (!best.hit) {
      const f32 t = 0.5f * (ray.dir.y + 1.f);
      const Vec3 sky = (1.f - t) * Vec3{1.f, 1.f, 1.f} + t * Vec3{0.5f, 0.7f, 1.f};
      radiance += throughput * sky;
      break;
    }

    const Vec3 target = best.p + best.n + random_unit_vector(rng);
    ray = Ray{best.p, normalize(target - best.p)};

    Vec3 albedo = spheres[0].albedo;
    if (Kokkos::fabs(best.n.y) > 0.9f)
      albedo = spheres[1].albedo;

    throughput = throughput * albedo;
  }

  return radiance;
}

PathTracer::PixelView PathTracer::render() const
{
  const u32 w = params.width;
  const u32 h = params.height;

  PixelView pixels("pixels", h, w);

  const f32 aspect = f32(w) / f32(h);
  const Camera cam = Camera::make_pinhole({3.f, 3.f, 2.f}, {0.f, 0.f, -1.f}, {0.f, 1.f, 0.f}, 20.f, aspect);

  const u32 spp = params.samples_per_pixel;

  Kokkos::parallel_for(
      "render",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(h), int(w)}),
      [=, self = *this](const int y, const int x) {
        PathTracer local = self;
        Rng rng(u32(1337u) ^ (u32(x) * 9781u) ^ (u32(y) * 6271u));
        Vec3 col{0.f, 0.f, 0.f};

        for (u32 s = 0; s < spp; ++s) {
          const f32 u = (f32(x) + rng.next_f32()) / f32(w - 1);
          const f32 v = (f32(y) + rng.next_f32()) / f32(h - 1);
          Ray ray = cam.ray(u, 1.f - v);
          col += local.shade(rng, ray);
        }

        const f32 inv = 1.f / f32(spp);
        pixels(y, x) = col * inv;
      });

  Kokkos::fence();
  return pixels;
}

} // namespace opencode::pt
