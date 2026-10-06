// Regression test for the OptiX instanced-UV drop: ObjectMesh -> TriangleMesh
// marshaling must carry UVs (and normals), not just positions/indices.
// Pure CPU-side test — no backend required.

#include "photon/pt/scene.h"

#include <Kokkos_Core.hpp>

#include <cassert>

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    using namespace photon::pt;

    {
      // Object with positions, indices, normals and UVs.
      ObjectMesh obj;
      obj.positions = {
          0.f, 0.f, 0.f,   1.f, 0.f, 0.f,   0.f, 1.f, 0.f,   // tri 0
          1.f, 0.f, 0.f,   2.f, 0.f, 0.f,   1.f, 1.f, 0.f};  // tri 1
      obj.indices = {0, 1, 2, 3, 4, 5};
      obj.normals = {
          0.f, 0.f, 1.f,   0.f, 0.f, 1.f,   0.f, 0.f, 1.f,
          0.f, 0.f, 1.f,   0.f, 0.f, 1.f,   0.f, 0.f, 1.f};
      obj.uvs = {
          0.f, 0.f,   1.f, 0.f,   0.f, 1.f,
          1.f, 0.f,   2.f, 0.f,   1.f, 1.f};
      obj.material_id = 7;

      TriangleMesh tm = object_mesh_to_triangle_mesh(obj);

      assert(tm.positions.extent(0) == 6);
      assert(tm.indices.extent(0) == 6);
      assert(tm.material_ids.extent(0) == 2);
      assert(tm.has_normals());
      // The actual regression: UVs must survive marshaling.
      assert(tm.has_texcoords());
      assert(tm.texcoords.extent(0) == 6);

      auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.positions);
      auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.indices);
      auto mat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.material_ids);
      auto nrm_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.normals);
      auto uv_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tm.texcoords);

      assert(pos_h(3).x == 1.f && pos_h(3).y == 0.f && pos_h(3).z == 0.f);
      assert(idx_h(3) == 3 && idx_h(5) == 5);
      assert(mat_h(0) == 7 && mat_h(1) == 7);
      assert(nrm_h(2).z == 1.f);
      assert(uv_h(4).x == 2.f && uv_h(4).y == 0.f);
      assert(uv_h(5).x == 1.f && uv_h(5).y == 1.f);
    }

    {
      // Object without UVs must produce an empty texcoords view (not garbage).
      ObjectMesh obj;
      obj.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
      obj.indices = {0, 1, 2};

      TriangleMesh tm = object_mesh_to_triangle_mesh(obj);
      assert(!tm.has_texcoords());
      assert(tm.positions.extent(0) == 3);
    }

    {
      // Object with normals but partial UVs: fewer UVs than 2 * vertices
      // must be treated as "no UVs" (same rule as the normals check).
      ObjectMesh obj;
      obj.positions = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
      obj.indices = {0, 1, 2};
      obj.uvs = {0.f, 0.f, 1.f, 0.f};  // incomplete

      TriangleMesh tm = object_mesh_to_triangle_mesh(obj);
      assert(!tm.has_texcoords());
    }
  }
  Kokkos::finalize();
  return 0;
}
