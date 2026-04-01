# Fix Environment Map Rotation — ✅ COMPLETE

## Problem
The environment map (sky dome) transform from the PBRT scene is parsed and stored in `pbrt.env_map_transform` but **never applied** during rendering. This caused the sky to appear with wrong orientation (visible seam, wrong sun position).

The Moana scene applies: `Scale -1 1 1`, `Rotate 90 -1 0 0`, `Rotate 65 0 0 1` before the `LightSource "infinite"` directive. These are accumulated into `env_map_transform` (column-major float[16]) by the parser.

## Changes Applied

### 1. `include/photon/pt/environment_map.h` — ✅ DONE

- ✅ Added `world_to_tex[9]` and `tex_to_world[9]` rotation matrix arrays (identity default)
- ✅ Added `to_tex_space()` and `to_world_space()` KOKKOS_FUNCTION helpers
- ✅ `evaluate()` uses `dir_to_uv(to_tex_space(direction))`
- ✅ `pdf()` uses `dir_to_uv(to_tex_space(direction))`
- ✅ `sample_direction()` uses `to_world_space(uv_to_dir(uv))`

### 2. `src/photon/pbrt/pbrt_to_photon.cpp` — ✅ DONE

- ✅ Rotation matrix extraction code inserted after `env.build_cdf()`, before `scene.env_map = env`
- ✅ Extracts 3×3 upper-left from `env_map_transform` into `world_to_tex` (inverse) and `tex_to_world` (forward)

## Verification

- Build: `[100%]` all targets, no compile errors
- Render: palmsCam scene rendered successfully in 41ms, 16spp
- Visual: Sky shows correct blue sky with clouds; palm trees visible; correct orientation
- Diagnostic: `Env map rotation applied (transform from scene)` confirmed in stderr

## Checklist
- [x] Add `world_to_tex[9]` and `tex_to_world[9]` arrays + helpers to `environment_map.h`
- [x] Modify `evaluate()` to transform direction to tex space before UV lookup
- [x] Modify `pdf()` to transform direction to tex space before UV lookup
- [x] Modify `sample_direction()` to transform sampled direction to world space
- [x] Add rotation matrix extraction code in `pbrt_to_photon.cpp`
- [x] Remove PTX cache, build, verify no compile errors
- [x] Render palmsCam and save as PNG
