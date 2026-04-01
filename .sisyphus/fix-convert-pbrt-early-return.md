# Fix: convert_pbrt_scene() early return skips materials/camera/envmap

## Problem
In `src/photon/pbrt/pbrt_to_photon.cpp`, `convert_pbrt_scene()` returns early at line 74-77 when `total_tris == 0`, skipping ALL material/camera/envmap setup. This breaks instanced-only scenes.

## Changes Required

### File: `src/photon/pbrt/pbrt_to_photon.cpp`

### Change 1: Remove early return (lines 74-77)

Replace:
```cpp
  if (total_tris == 0) {
    std::fprintf(stderr, "pbrt scene has no triangles\n");
    return result;
  }

  const uint64_t total_verts = total_tris * 3;
```

With:
```cpp
  if (total_tris == 0) {
    std::fprintf(stderr, "pbrt scene has no scene-level triangles (instanced-only scene)\n");
  }
```

### Change 2: Move `Scene &scene` before mesh block, wrap mesh in conditional

Currently around line 278: `Scene &scene = result.scene;`

Move that BEFORE the mesh allocation block. Then wrap the mesh block in `if (total_tris > 0) { ... }`.

The result:

```cpp
  Scene &scene = result.scene;

  if (total_tris > 0) {
    const uint64_t total_verts = total_tris * 3;

    TriangleMesh tm;
    tm.positions = Kokkos::View<Vec3 *>("pos", total_verts);
    tm.indices = Kokkos::View<u32 *>("idx", total_verts);
    tm.material_ids = Kokkos::View<u32 *>("mat_id", total_tris);
    tm.albedo_per_prim = Kokkos::View<Vec3 *>("alb", total_tris);
    tm.normals = Kokkos::View<Vec3 *>("normals", total_verts);
    tm.texcoords = Kokkos::View<Vec2 *>("uv", total_verts);

    auto pos_h = Kokkos::create_mirror_view(tm.positions);
    auto idx_h = Kokkos::create_mirror_view(tm.indices);
    auto mat_id_h = Kokkos::create_mirror_view(tm.material_ids);
    auto alb_h = Kokkos::create_mirror_view(tm.albedo_per_prim);
    auto nrm_h = Kokkos::create_mirror_view(tm.normals);
    auto uv_h = Kokkos::create_mirror_view(tm.texcoords);

    uint64_t vert_off = 0;
    uint64_t tri_off = 0;

    for (const auto &mesh : pbrt.meshes) {
      // ... entire existing mesh processing loop ...
    }

    Kokkos::deep_copy(tm.positions, pos_h);
    Kokkos::deep_copy(tm.indices, idx_h);
    Kokkos::deep_copy(tm.material_ids, mat_id_h);
    Kokkos::deep_copy(tm.albedo_per_prim, alb_h);
    Kokkos::deep_copy(tm.normals, nrm_h);
    Kokkos::deep_copy(tm.texcoords, uv_h);

    scene.mesh = tm;
    scene.bvh = Bvh::build_cpu(tm);
  }
```

### What stays OUTSIDE the conditional (always runs):
- materials_cpu, lights_cpu, emissive vectors declarations
- mat_name_to_id construction from named_materials
- Material upload to scene.materials
- Texture upload
- Emissive/light setup (will no-op if no emissive prims)
- Environment map loading
- Camera setup
- result.mat_name_to_id = mat_name_to_id

## Build
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake --build /home/mvictoras/src/photon/build -j$(nproc)
```

## Test
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/home/mvictoras/src/photon/build:/home/mvictoras/src/photon/build/_deps/icet-build/lib:/home/mvictoras/src/embree-install/lib:/home/mvictoras/src/kokkos-cuda-install/lib:/opt/hfs21.0.596/dsolib:/home/mvictoras/src/openmpi-cuda-install/lib
PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /storage/moana/island/pbrt-v4/island_palmsCam_only.pbrt -o /home/mvictoras/src/photon/renders/moana_palmsCam_only.ppm --spp 16 --width 960 --height 402 --exposure 1.0 --use-ias
```

Expected: non-zero materials, camera at correct position, geometry visible.
