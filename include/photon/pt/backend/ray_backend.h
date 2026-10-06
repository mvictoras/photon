#pragma once

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <memory>

#include "photon/pt/math.h"
#include "photon/pt/math_vec2.h"
#include "photon/pt/scene.h"

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

  // Whether this backend builds true two-level acceleration structures
  // (e.g. OptiX IAS). If false, build_accel_instanced() falls back to
  // CPU-side flattening so instanced geometry is still rendered.
  virtual bool supports_instancing() const { return false; }

  // Handle a scene with instanced geometry. The default implementation
  // flattens the instances into a single flat TriangleMesh on the CPU
  // (never silently dropping geometry) and builds the flat path. Backends
  // with native two-level BVH support override this.
  virtual void build_accel_instanced(const Scene &scene,
                                     const InstancedGeometry &instanced) {
    if (instanced.empty()) {
      build_accel(scene);
      return;
    }
    Scene flat = scene;
    flat.mesh = flatten_instanced_geometry(scene, instanced);
    flat.bvh = Bvh::build_cpu(flat.mesh);
    std::fprintf(stderr,
        "[photon] %s: no IAS support — flattened %zu instances into %u triangles\n",
        name(), instanced.instances.size(), flat.mesh.triangle_count());
    build_accel(flat);
  }

  virtual void trace_closest(const RayBatch &rays, HitBatch &hits) = 0;
  virtual void trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded) = 0;
  virtual const char *name() const = 0;
};

std::unique_ptr<RayBackend> create_best_backend();

}
