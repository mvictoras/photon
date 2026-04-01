#pragma once

#include <Kokkos_Core.hpp>

#include <memory>

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"

namespace photon::pt {

struct Scene;
struct InstancedGeometry;

struct HitResult {
  f32 t{0.f};
  Vec3 position;
  Vec3 normal;
  Vec3 shading_normal;
  Vec2 uv;
  Vec3 interpolated_color{0.f, 0.f, 0.f};
  u32 prim_id{0};
  u32 geom_id{0};
  u32 inst_id{0};
  u32 material_id{0};
  bool hit{false};
  bool has_interpolated_color{false};
};

struct RayBatch {
  Kokkos::View<Vec3 *> origins;
  Kokkos::View<Vec3 *> directions;
  Kokkos::View<f32 *> tmin;
  Kokkos::View<f32 *> tmax;
  u32 count{0};
};

struct HitBatch {
  Kokkos::View<HitResult *> hits;
  u32 count{0};
};

struct RayBackend {
  virtual ~RayBackend() = default;
  virtual void build_accel(const Scene &scene) = 0;
  virtual void build_accel_instanced(const Scene &scene,
                                      const InstancedGeometry & /*instanced*/) {
    build_accel(scene);
  }
  virtual void trace_closest(const RayBatch &rays, HitBatch &hits) = 0;
  virtual void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded) = 0;
  virtual const char *name() const = 0;
};

std::unique_ptr<RayBackend> create_best_backend();

}
