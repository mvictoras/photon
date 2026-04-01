# Architecture Refactor: Decouple OptixBackend from PBRT Types

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the reverse dependency where `OptixBackend` (core renderer) includes PBRT file-format types, merge vestigial `pathtracer/` directory into `pt/`, and simplify app-level instancing code.

**Architecture:** Define format-agnostic instancing types (`ObjectMesh`, `InstancedGeometry`) in `pt/scene.h`. Move instance preparation from the app into `pbrt_to_photon.cpp`. Add a virtual `build_accel_instanced()` to `RayBackend` so apps don't need `dynamic_cast` or `goto`. Merge `pathtracer/version.{h,cpp}` into `pt/version.{h,cpp}`.

**Tech Stack:** C++20, Kokkos, OptiX 8, CMake

**Build command:** `cmake --build /home/mvictoras/src/photon/build -j$(nproc)`

**Render test (bathroom):** `PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /home/mvictoras/src/photon/scenes/bathroom/scene-v4.pbrt -o /tmp/test_bathroom.ppm --spp 4 --width 256 --height 256`

**Render test (Moana IAS):** `PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /storage/moana/island/pbrt-v4/island_palmsCam_only.pbrt -o /tmp/test_moana.ppm --spp 4 --width 256 --height 256 --use-ias`

**Environment:**
```bash
export PATH=/usr/local/cuda-12.8/bin:/home/mvictoras/src/openmpi-cuda-install/bin:$PATH
export LD_LIBRARY_PATH=/home/mvictoras/src/photon/build:/home/mvictoras/src/photon/build/_deps/icet-build/lib:/home/mvictoras/src/embree-install/lib:/home/mvictoras/src/kokkos-cuda-install/lib:/opt/hfs21.0.596/dsolib:/home/mvictoras/src/openmpi-cuda-install/lib
```

---

## Task 1: Add format-agnostic instancing types to scene.h

**Files:**
- Modify: `include/photon/pt/scene.h`

**Context:** There is already an unused `Instance` struct at line 15-20 of `scene.h`. We need to repurpose it and add `ObjectMesh` and `InstancedGeometry`.

**Step 1: Replace the unused Instance struct and add new types**

In `include/photon/pt/scene.h`, replace the existing `Instance` struct (lines 15-20) and add new types before the `Scene` struct. The file needs `<vector>` and `<cstdint>` includes added.

Replace:
```cpp
struct Instance {
  Mat4 transform;
  Mat4 inv_transform;
  u32 mesh_id{0};
  u32 material_id{0};
};
```

With:
```cpp
// Format-agnostic mesh data for instanced acceleration structures.
// Used by backends that support two-level BVH (e.g. OptiX IAS).
// Scene converters (pbrt_to_photon, SceneFromAnari) populate these;
// backends consume them without knowing the source format.
struct ObjectMesh {
  std::vector<float> positions;   // xyz interleaved
  std::vector<int> indices;
  std::vector<float> normals;     // xyz interleaved (may be empty)
  std::vector<float> uvs;         // uv interleaved (may be empty)
  u32 material_id{0};
};

struct Instance {
  u32 object_id{0};
  float transform[16]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};

struct InstancedGeometry {
  std::vector<ObjectMesh> objects;   // one per unique object
  std::vector<Instance> instances;   // references objects by index
  bool empty() const { return objects.empty() || instances.empty(); }
};
```

Add `#include <vector>` to the includes at the top of `scene.h`.

**Step 2: Verify build compiles**

Run: `cmake --build /home/mvictoras/src/photon/build -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds (no code uses the old `Instance` struct).

**Step 3: Commit**

```bash
git add include/photon/pt/scene.h
git commit -m "refactor: add format-agnostic instancing types to scene.h

Add ObjectMesh, Instance, and InstancedGeometry structs that decouple
the instanced acceleration structure building from any specific file
format (PBRT, ANARI, etc.). Replace the unused Instance struct."
```

---

## Task 2: Add virtual build_accel_instanced to RayBackend

**Files:**
- Modify: `include/photon/pt/backend/ray_backend.h`

**Step 1: Add forward declaration and virtual method**

In `ray_backend.h`, add a forward declaration for `InstancedGeometry` alongside the existing `struct Scene;` (line 12), and add a virtual method to `RayBackend` after `build_accel`:

After `struct Scene;` add:
```cpp
struct InstancedGeometry;
```

After `virtual void build_accel(const Scene &scene) = 0;` (line 44) add:
```cpp
virtual void build_accel_instanced(const Scene &scene,
                                    const InstancedGeometry &instanced) {
  // Default: ignore instancing, fall back to flat scene
  build_accel(scene);
}
```

**Step 2: Verify build compiles**

Run: `cmake --build /home/mvictoras/src/photon/build -j$(nproc) 2>&1 | tail -5`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add include/photon/pt/backend/ray_backend.h
git commit -m "refactor: add virtual build_accel_instanced to RayBackend

Provides a default implementation that falls back to build_accel(),
so only backends that support instancing need to override it."
```

---

## Task 3: Convert OptixBackend to use format-agnostic types

**Files:**
- Modify: `include/photon/pt/backend/optix_backend.h`
- Modify: `src/photon/pt/backend/optix_backend.cpp`

**Step 1: Update optix_backend.h**

Remove the PBRT include and change signatures:

1. Remove line 4: `#include "photon/pbrt/pbrt_scene.h"`
2. Add: `#include "photon/pt/scene.h"` (needed for InstancedGeometry)
3. Remove `#include <map>` and `#include <string>` (no longer needed after removing pbrt types)

4. Change `build_accel_instanced` signature (lines 21-24) from:
```cpp
void build_accel_instanced(const Scene &scene,
                            const std::vector<photon::pbrt::PbrtTriMesh> *object_meshes_ptr,
                            const std::vector<photon::pbrt::PbrtInstance> *instances_ptr,
                            const std::map<std::string, u32> *mat_name_to_id);
```
To:
```cpp
void build_accel_instanced(const Scene &scene,
                            const InstancedGeometry &instanced) override;
```

5. Change `build_ias` signature (line 83) from:
```cpp
void build_ias(const std::vector<ObjectGAS> &gas_list, const std::vector<photon::pbrt::PbrtInstance> &instances, const ObjectGAS *scene_gas);
```
To:
```cpp
void build_ias(const std::vector<ObjectGAS> &gas_list,
               const std::vector<Instance> &instances,
               const ObjectGAS *scene_gas);
```

**Step 2: Update optix_backend.cpp — build_ias method**

The `build_ias` method (around line 499) currently takes `std::vector<photon::pbrt::PbrtInstance>` and reads `inst.object_name` and `inst.transform`.

Change signature to:
```cpp
void OptixBackend::build_ias(const std::vector<ObjectGAS> &gas_list,
                              const std::vector<Instance> &instances,
                              const ObjectGAS *scene_gas)
```

Inside the method body, replace the name-based lookup with index-based lookup. The current code (lines 507-514) builds a `name_to_idx` map from instance object names. Replace that entire block and the loop that uses it.

Replace lines 507-533 (the name_to_idx mapping and per_obj_instances building) with:
```cpp
  std::vector<std::vector<OptixInstance>> per_obj_instances(n_obj);
  for (const auto &inst : instances) {
    const u32 obj_idx = inst.object_id;
    if (obj_idx >= n_obj || gas_list[obj_idx].handle == 0) continue;

    Mat4 xfm = col_major_to_mat4(inst.transform);
    OptixInstance oi{};
    oi.instanceId      = obj_idx;
    oi.sbtOffset       = obj_idx;
    oi.visibilityMask  = 0xFF;
    oi.flags           = OPTIX_INSTANCE_FLAG_NONE;
    oi.traversableHandle = gas_list[obj_idx].handle;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        oi.transform[r*4+c] = xfm.m[r][c];
    per_obj_instances[obj_idx].push_back(oi);
  }
```

**Step 3: Update optix_backend.cpp — build_accel_instanced method**

The `build_accel_instanced` method (around line 664) currently takes PBRT types and does name-based grouping. Replace the entire method with index-based logic using `InstancedGeometry`.

Replace the entire method with:
```cpp
void OptixBackend::build_accel_instanced(const Scene &scene,
    const InstancedGeometry &instanced)
{
  if (instanced.empty()) {
    build_accel(scene);
    return;
  }

  if (!m_pipeline) create_pipeline();

  const u32 n_objects = u32(instanced.objects.size());
  m_object_gas.clear();
  m_object_gas.resize(n_objects);

  for (u32 oi = 0; oi < n_objects; ++oi) {
    const auto &obj = instanced.objects[oi];
    if (obj.indices.empty()) continue;

    const uint64_t nv = obj.positions.size() / 3;
    const uint64_t nt = obj.indices.size() / 3;
    const bool has_normals = obj.normals.size() >= obj.positions.size();

    TriangleMesh tm;
    tm.positions    = Kokkos::View<Vec3*>("p", nv);
    tm.indices      = Kokkos::View<u32*>("i", nt * 3);
    tm.material_ids = Kokkos::View<u32*>("m", nt);
    if (has_normals)
      tm.normals    = Kokkos::View<Vec3*>("n", nv);

    auto ph = Kokkos::create_mirror_view(tm.positions);
    auto ih = Kokkos::create_mirror_view(tm.indices);
    auto mh = Kokkos::create_mirror_view(tm.material_ids);
    auto nh = has_normals ? Kokkos::create_mirror_view(tm.normals)
                          : decltype(Kokkos::create_mirror_view(tm.normals)){};

    for (uint64_t v = 0; v < nv; ++v)
      ph(v) = {obj.positions[v*3], obj.positions[v*3+1], obj.positions[v*3+2]};
    for (uint64_t i = 0; i < nt * 3; ++i)
      ih(i) = u32(obj.indices[i]);
    for (uint64_t t = 0; t < nt; ++t)
      mh(t) = obj.material_id;
    if (has_normals)
      for (uint64_t v = 0; v < nv; ++v)
        nh(v) = {obj.normals[v*3], obj.normals[v*3+1], obj.normals[v*3+2]};

    Kokkos::deep_copy(tm.positions, ph);
    Kokkos::deep_copy(tm.indices,   ih);
    Kokkos::deep_copy(tm.material_ids, mh);
    if (has_normals) Kokkos::deep_copy(tm.normals, nh);
    build_gas_for_mesh(tm, m_object_gas[oi]);
  }

  ObjectGAS scene_gas{};
  bool has_scene_mesh = scene.mesh.triangle_count() > 0;
  if (has_scene_mesh) {
    build_gas_for_mesh(scene.mesh, scene_gas);
    m_mesh = scene.mesh;
  }

  build_ias(m_object_gas, instanced.instances, has_scene_mesh ? &scene_gas : nullptr);
  std::fprintf(stderr, "[photon] OptiX IAS: %u objects, %zu instances\n",
      n_objects, instanced.instances.size());
}
```

**Important notes for the implementer:**
- The old code grouped multiple `PbrtTriMesh` by `object_name` into one GAS. The new code expects each `ObjectMesh` to already be a single merged mesh per object. The merging will be done by `pbrt_to_photon.cpp` in Task 4.
- The old code looked up `material_name` in `mat_name_to_id` map per mesh. The new code uses `obj.material_id` directly (resolved during conversion).

**Step 4: Verify build compiles**

Run: `cmake --build /home/mvictoras/src/photon/build -j$(nproc) 2>&1 | tail -20`

Note: This will cause build errors in `photon_pbrt_render.cpp` because the app still calls the old API. That's expected — Task 4 and 5 fix the app.

Actually, to keep the build green between tasks, we should do Task 4 before or together with this one. See note below.

**Step 5: Commit**

```bash
git add include/photon/pt/backend/optix_backend.h src/photon/pt/backend/optix_backend.cpp
git commit -m "refactor: decouple OptixBackend from PBRT types

Use format-agnostic ObjectMesh/Instance/InstancedGeometry types
instead of PbrtTriMesh/PbrtInstance. Remove #include pbrt_scene.h
from optix_backend.h, eliminating the reverse dependency."
```

---

## Task 4: Move instance preparation into pbrt_to_photon

**Files:**
- Modify: `include/photon/pbrt/pbrt_to_photon.h`
- Modify: `src/photon/pbrt/pbrt_to_photon.cpp`

**Step 1: Update ConvertedScene in pbrt_to_photon.h**

Change the `ConvertedScene` struct to include `InstancedGeometry` and remove `mat_name_to_id`:

```cpp
#pragma once

#include "photon/pbrt/pbrt_scene.h"
#include "photon/pt/camera.h"
#include "photon/pt/scene.h"
#include <map>
#include <string>

namespace photon::pbrt {

struct ConvertedScene {
  photon::pt::Scene scene;
  photon::pt::Camera camera;
  photon::pt::InstancedGeometry instanced_geometry;
};

ConvertedScene convert_pbrt_scene(const PbrtScene &pbrt, const std::string &base_dir = "");

}
```

**Step 2: Update pbrt_to_photon.cpp — build InstancedGeometry**

At the end of `convert_pbrt_scene()`, after building the scene and camera, add code to populate `result.instanced_geometry` from the PBRT data. This is essentially the logic currently in `photon_pbrt_render.cpp` lines 152-160, but done properly inside the converter where we have access to `mat_name_to_id`.

Find the code near the end of `convert_pbrt_scene()` that builds `result` and add before the return:

```cpp
  // Build format-agnostic instanced geometry for backends that support IAS.
  if (!pbrt.object_defs.empty() && !pbrt.object_instances.empty()) {
    auto &ig = result.instanced_geometry;
    
    // Map object names to indices
    std::map<std::string, u32> obj_name_to_idx;
    for (const auto &[name, meshes] : pbrt.object_defs) {
      u32 idx = u32(ig.objects.size());
      obj_name_to_idx[name] = idx;
      
      // Merge all sub-meshes of this object into one ObjectMesh
      // (PBRT objects can have multiple Shape definitions)
      photon::pt::ObjectMesh obj;
      uint64_t voff = 0;
      for (const auto &m : meshes) {
        const uint64_t nv = m.positions.size() / 3;
        
        // Resolve material
        u32 mat_id = 0;
        auto it = mat_name_to_id.find(m.material_name);
        if (it != mat_name_to_id.end()) mat_id = it->second;
        obj.material_id = mat_id;  // last sub-mesh wins (usually all same)
        
        // Append positions
        obj.positions.insert(obj.positions.end(),
            m.positions.begin(), m.positions.end());
        
        // Append indices (offset by accumulated vertex count)
        for (int idx_val : m.indices)
          obj.indices.push_back(idx_val + int(voff));
        
        // Append normals
        if (m.normals.size() >= m.positions.size())
          obj.normals.insert(obj.normals.end(),
              m.normals.begin(), m.normals.end());
        
        // Append UVs
        if (!m.uvs.empty())
          obj.uvs.insert(obj.uvs.end(), m.uvs.begin(), m.uvs.end());
        
        voff += nv;
      }
      ig.objects.push_back(std::move(obj));
    }
    
    // Convert instances
    for (const auto &inst : pbrt.object_instances) {
      auto it = obj_name_to_idx.find(inst.object_name);
      if (it == obj_name_to_idx.end()) continue;
      
      photon::pt::Instance gi;
      gi.object_id = it->second;
      std::memcpy(gi.transform, inst.transform, sizeof(float) * 16);
      ig.instances.push_back(gi);
    }
  }
```

Make sure `#include <cstring>` is present for `std::memcpy`.

**Important:** The `mat_name_to_id` map is a local variable inside `convert_pbrt_scene()`. After this change it no longer needs to be in `ConvertedScene`. It's used internally during conversion and also used to populate `ObjectMesh::material_id`.

**Step 3: Verify build compiles**

Run: `cmake --build /home/mvictoras/src/photon/build -j$(nproc) 2>&1 | tail -20`

There will be build errors in `photon_pbrt_render.cpp` because it still references `converted.mat_name_to_id`. That's fixed in Task 5.

**Step 4: Commit (combine with Task 5 if build doesn't pass independently)**

```bash
git add include/photon/pbrt/pbrt_to_photon.h src/photon/pbrt/pbrt_to_photon.cpp
git commit -m "refactor: build InstancedGeometry inside pbrt_to_photon

Move instance flattening and material ID resolution from the app into
the converter. ConvertedScene now includes InstancedGeometry instead
of mat_name_to_id."
```

---

## Task 5: Simplify app code

**Files:**
- Modify: `app/photon_pbrt_render.cpp`
- Modify: `app/photon_mpi_render.cpp` (if it uses IAS — check first)
- Modify: `app/photon_bdpt_render.cpp` (if it uses IAS — check first)

**Step 1: Simplify photon_pbrt_render.cpp**

Remove the `#include "photon/pt/backend/optix_backend.h"` (line 14). The app no longer needs to know about OptixBackend directly.

Replace the entire `#ifdef PHOTON_HAS_OPTIX` block (lines 148-172) and the `backend->build_accel` call with:

```cpp
    if (!converted.instanced_geometry.empty()) {
      std::fprintf(stderr, "Using IAS: %zu objects, %zu instances\n",
          converted.instanced_geometry.objects.size(),
          converted.instanced_geometry.instances.size());
    }
    backend->build_accel_instanced(converted.scene, converted.instanced_geometry);
```

Note: When `instanced_geometry` is empty, `build_accel_instanced()` falls back to `build_accel()` via the default virtual method in `RayBackend`.

Also remove any reference to `converted.mat_name_to_id` since it no longer exists.

**Step 2: Check photon_mpi_render.cpp and photon_bdpt_render.cpp**

Check if either file references `mat_name_to_id`, `PbrtTriMesh`, `PbrtInstance`, or `OptixBackend`. If so, apply the same simplification. If not, no changes needed.

**Step 3: Build and test**

```bash
cmake --build /home/mvictoras/src/photon/build -j$(nproc)
```
Expected: Clean build, no errors.

Test with bathroom scene (non-IAS path):
```bash
PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /home/mvictoras/src/photon/scenes/bathroom/scene-v4.pbrt -o /tmp/test_bathroom.ppm --spp 4 --width 256 --height 256
```
Expected: Renders successfully.

Test with Moana scene (IAS path):
```bash
PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /storage/moana/island/pbrt-v4/island_palmsCam_only.pbrt -o /tmp/test_moana.ppm --spp 4 --width 256 --height 256 --use-ias
```
Expected: Renders successfully with IAS.

**Step 4: Commit**

```bash
git add app/photon_pbrt_render.cpp
# Also add mpi/bdpt if changed
git commit -m "refactor: simplify app instancing code

Remove dynamic_cast, goto, and #ifdef PHOTON_HAS_OPTIX from
photon_pbrt_render. Use the virtual build_accel_instanced() method
which falls back to build_accel() for non-instanced scenes."
```

---

## Task 6: Merge pathtracer/ into pt/

**Files:**
- Move: `include/photon/pathtracer/version.h` → `include/photon/pt/version.h`
- Move: `src/photon/pathtracer/version.cpp` → `src/photon/pt/version.cpp`
- Modify: `app/main.cpp` (update include)
- Modify: `tests/version_test.cpp` (update include)
- Modify: `src/CMakeLists.txt` (update source paths)
- Delete: empty `include/photon/pathtracer/` and `src/photon/pathtracer/` directories

**Step 1: Move files using git mv**

```bash
git mv include/photon/pathtracer/version.h include/photon/pt/version.h
git mv src/photon/pathtracer/version.cpp src/photon/pt/version.cpp
rmdir include/photon/pathtracer
rmdir src/photon/pathtracer
```

**Step 2: Update includes**

In `app/main.cpp`, change:
```cpp
#include "photon/pathtracer/version.h"
```
To:
```cpp
#include "photon/pt/version.h"
```

In `tests/version_test.cpp`, make the same change.

In `src/photon/pt/version.cpp`, change:
```cpp
#include "photon/pathtracer/version.h"
```
To:
```cpp
#include "photon/pt/version.h"
```

**Step 3: Update CMakeLists.txt**

In `src/CMakeLists.txt`, change line 48:
```
photon/pathtracer/version.cpp
```
To:
```
photon/pt/version.cpp
```

And change line 84:
```
${PROJECT_SOURCE_DIR}/include/photon/pathtracer/version.h
```
To:
```
${PROJECT_SOURCE_DIR}/include/photon/pt/version.h
```

**Step 4: Build and test**

```bash
cmake --build /home/mvictoras/src/photon/build -j$(nproc)
```
Expected: Clean build.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: merge pathtracer/ into pt/

Move version.h and version.cpp from the near-empty pathtracer/
directory into pt/ where the actual pathtracer code lives.
Eliminates confusing duplicate naming."
```

---

## Task 7: Final verification

**Step 1: Clean build**

```bash
cmake --build /home/mvictoras/src/photon/build -j$(nproc) 2>&1 | tail -5
```

**Step 2: Verify no PBRT references in pt/**

```bash
grep -r "photon/pbrt" include/photon/pt/ src/photon/pt/
```
Expected: No matches.

**Step 3: Render tests**

Bathroom (non-IAS):
```bash
PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /home/mvictoras/src/photon/scenes/bathroom/scene-v4.pbrt -o /tmp/test_bathroom.ppm --spp 4 --width 256 --height 256
```

Moana palmsCam (IAS):
```bash
PHOTON_BACKEND=optix /home/mvictoras/src/photon/build/_deps/icet-build/bin/photon_pbrt_render /storage/moana/island/pbrt-v4/island_palmsCam_only.pbrt -o /tmp/test_moana.ppm --spp 4 --width 256 --height 256 --use-ias
```

Both should render without errors.

---

## Implementation Note: Task Ordering

Tasks 3, 4, and 5 must be done together (or at least committed together) because they change an API boundary:
- Task 3 changes OptixBackend's API
- Task 4 changes ConvertedScene's API
- Task 5 updates the app to match

If you want the build to stay green between commits, do Tasks 3+4+5 as a single commit. Alternatively, you can temporarily keep both the old and new APIs in OptixBackend until all callers are updated.

Recommended approach: Do Tasks 3+4+5 as one batch, then commit with a combined message. Tasks 1, 2, and 6 can be independent commits.
