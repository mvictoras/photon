#pragma once

#include <Kokkos_Core.hpp>

#include "photon/pt/light.h"
#include "photon/pt/material.h"
#include "photon/pt/texture.h"

#include <vector>

namespace photon::pt {

// GPU-upload boilerplate shared by scene converters (PBRT, ANARI, and any
// future format). Each helper wraps the create_mirror_view / deep_copy dance
// that was previously hand-written in both converters. Import logic itself
// stays per-format — only the upload step is shared.

// Host materials -> device view. Caller sets Scene::material_count.
Kokkos::View<Material *> upload_materials(const std::vector<Material> &materials);

// Host lights -> device view. Caller sets Scene::light_count.
Kokkos::View<Light *> upload_lights(const std::vector<Light> &lights);

// Host emissive-primitive ids/areas -> device views.
Kokkos::View<u32 *> upload_u32(const std::vector<u32> &values);
Kokkos::View<f32 *> upload_f32(const std::vector<f32> &values);

// One image's texels for the atlas: tightly packed interleaved RGB floats.
struct AtlasImage {
  const float *rgb;  // width * height * 3 floats
  u32 width;
  u32 height;
};

// Images (in atlas order) -> flat TextureAtlas with one TextureInfo entry
// per image. Returns a zero-count atlas for an empty input.
TextureAtlas upload_texture_atlas(const std::vector<AtlasImage> &images);

} // namespace photon::pt
