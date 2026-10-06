#include "photon/pt/scene/upload.h"

namespace photon::pt {

Kokkos::View<Material *> upload_materials(const std::vector<Material> &materials)
{
  const u32 count = u32(materials.size());
  Kokkos::View<Material *> out("materials", count);
  auto host = Kokkos::create_mirror_view(out);
  for (u32 i = 0; i < count; ++i)
    host(i) = materials[i];
  Kokkos::deep_copy(out, host);
  return out;
}

Kokkos::View<Light *> upload_lights(const std::vector<Light> &lights)
{
  const u32 count = u32(lights.size());
  Kokkos::View<Light *> out("lights", count);
  auto host = Kokkos::create_mirror_view(out);
  for (u32 i = 0; i < count; ++i)
    host(i) = lights[i];
  Kokkos::deep_copy(out, host);
  return out;
}

Kokkos::View<u32 *> upload_u32(const std::vector<u32> &values)
{
  const u32 count = u32(values.size());
  Kokkos::View<u32 *> out("u32_values", count);
  auto host = Kokkos::create_mirror_view(out);
  for (u32 i = 0; i < count; ++i)
    host(i) = values[i];
  Kokkos::deep_copy(out, host);
  return out;
}

Kokkos::View<f32 *> upload_f32(const std::vector<f32> &values)
{
  const u32 count = u32(values.size());
  Kokkos::View<f32 *> out("f32_values", count);
  auto host = Kokkos::create_mirror_view(out);
  for (u32 i = 0; i < count; ++i)
    host(i) = values[i];
  Kokkos::deep_copy(out, host);
  return out;
}

TextureAtlas upload_texture_atlas(const std::vector<AtlasImage> &images)
{
  TextureAtlas atlas;
  if (images.empty())
    return atlas;

  atlas.count = u32(images.size());

  size_t total_pixels = 0;
  for (const auto &img : images)
    total_pixels += size_t(img.width) * size_t(img.height);

  atlas.pixels = Kokkos::View<Vec3 *>("atlas_pixels", total_pixels);
  atlas.infos = Kokkos::View<TextureInfo *>("atlas_infos", atlas.count);

  auto pixels_h = Kokkos::create_mirror_view(atlas.pixels);
  auto infos_h = Kokkos::create_mirror_view(atlas.infos);

  u32 pixel_offset = 0;
  for (u32 ti = 0; ti < atlas.count; ++ti) {
    const AtlasImage &img = images[ti];
    infos_h(ti).offset = pixel_offset;
    infos_h(ti).width = img.width;
    infos_h(ti).height = img.height;

    const u32 npixels = img.width * img.height;
    for (u32 p = 0; p < npixels; ++p)
      pixels_h(pixel_offset + p) = {img.rgb[p * 3], img.rgb[p * 3 + 1], img.rgb[p * 3 + 2]};
    pixel_offset += npixels;
  }

  Kokkos::deep_copy(atlas.pixels, pixels_h);
  Kokkos::deep_copy(atlas.infos, infos_h);
  return atlas;
}

} // namespace photon::pt
