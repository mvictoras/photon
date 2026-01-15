#include "opencode/pt/scene/builder.h"

namespace opencode::pt {

Scene SceneBuilder::make_two_quads()
{
  TriangleMesh mesh;

  mesh.positions = Kokkos::View<Vec3 *>("pos", 8);
  mesh.indices = Kokkos::View<u32 *>("idx", 12);
  mesh.albedo_per_prim = Kokkos::View<Vec3 *>("alb", 4);

  auto pos_h = Kokkos::create_mirror_view(mesh.positions);
  auto idx_h = Kokkos::create_mirror_view(mesh.indices);
  auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);

  // Quad A (vertical)
  pos_h(0) = {-1.f, -1.f, -2.5f};
  pos_h(1) = {+1.f, -1.f, -2.5f};
  pos_h(2) = {+1.f, +1.f, -2.5f};
  pos_h(3) = {-1.f, +1.f, -2.5f};

  // Quad B (ground)
  pos_h(4) = {-2.f, -1.f, -1.5f};
  pos_h(5) = {+2.f, -1.f, -1.5f};
  pos_h(6) = {+2.f, -1.f, -5.5f};
  pos_h(7) = {-2.f, -1.f, -5.5f};

  // indices: 4 triangles
  idx_h(0) = 0;
  idx_h(1) = 1;
  idx_h(2) = 2;
  idx_h(3) = 0;
  idx_h(4) = 2;
  idx_h(5) = 3;

  idx_h(6) = 4;
  idx_h(7) = 5;
  idx_h(8) = 6;
  idx_h(9) = 4;
  idx_h(10) = 6;
  idx_h(11) = 7;

  alb_h(0) = {0.9f, 0.2f, 0.2f};
  alb_h(1) = {0.2f, 0.9f, 0.2f};
  alb_h(2) = {0.7f, 0.7f, 0.7f};
  alb_h(3) = {0.7f, 0.7f, 0.7f};

  Kokkos::deep_copy(mesh.positions, pos_h);
  Kokkos::deep_copy(mesh.indices, idx_h);
  Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);

  Scene s;
  s.mesh = mesh;
  s.bvh = Bvh::build_cpu(mesh);
  return s;
}

} // namespace opencode::pt
