#include <anari/anari.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#include "photon/pt/pathtracer.h"
#include "photon/pt/scene/builder.h"

#include "photon/anari/SceneFromAnari.h"

namespace photon::anari_device {

namespace {

void owned_memory_deleter(const void *, const void *mem)
{
  delete[] static_cast<const char *>(mem);
}

size_t sizeof_anari(ANARIDataType type)
{
  switch (type) {
  case ANARI_INT32:
    return 4;
  case ANARI_UINT32:
    return 4;
  case ANARI_FLOAT32:
    return 4;
  case ANARI_FLOAT32_VEC2:
    return 8;
  case ANARI_FLOAT32_VEC3:
    return 12;
  case ANARI_FLOAT32_VEC4:
    return 16;
  case ANARI_UINT32_VEC2:
    return 8;
  case ANARI_UINT32_VEC3:
    return 12;
  case ANARI_UINT32_VEC4:
    return 16;
  default:
    return 0;
  }
}

size_t bytes_for(ANARIDataType type, uint64_t n1, uint64_t n2 = 1, uint64_t n3 = 1)
{
  return sizeof_anari(type) * size_t(n1) * size_t(n2) * size_t(n3);
}

}

PhotonDevice::PhotonDevice(ANARILibrary) {}

uintptr_t PhotonDevice::alloc_handle(ANARIDataType t)
{
  const uintptr_t h = uintptr_t(m_next_id++);
  m_objects[h] = std::make_unique<Object>();
  m_objects[h]->object_type = t;
  return h;
}

PhotonDevice::Object *PhotonDevice::get(ANARIObject o)
{
  auto it = m_objects.find((uintptr_t)o);
  return it == m_objects.end() ? nullptr : it->second.get();
}

ANARIArray1D PhotonDevice::newArray1D(
    const void *appMemory, ANARIMemoryDeleter deleter, const void *userData, ANARIDataType type, uint64_t n1)
{
  const uintptr_t h = alloc_handle(ANARI_ARRAY1D);
  auto *o = get((ANARIObject)h);
  o->memory = appMemory ? appMemory : (const void *)new char[bytes_for(type, n1)];
  o->deleter = appMemory ? deleter : owned_memory_deleter;
  o->userdata = appMemory ? userData : nullptr;
  return (ANARIArray1D)h;
}

ANARIArray2D PhotonDevice::newArray2D(
    const void *appMemory, ANARIMemoryDeleter deleter, const void *userData, ANARIDataType type, uint64_t n1, uint64_t n2)
{
  const uintptr_t h = alloc_handle(ANARI_ARRAY2D);
  auto *o = get((ANARIObject)h);
  o->memory = appMemory ? appMemory : (const void *)new char[bytes_for(type, n1, n2)];
  o->deleter = appMemory ? deleter : owned_memory_deleter;
  o->userdata = appMemory ? userData : nullptr;
  return (ANARIArray2D)h;
}

ANARIArray3D PhotonDevice::newArray3D(const void *appMemory, ANARIMemoryDeleter deleter, const void *userData,
    ANARIDataType type, uint64_t n1, uint64_t n2, uint64_t n3)
{
  const uintptr_t h = alloc_handle(ANARI_ARRAY3D);
  auto *o = get((ANARIObject)h);
  o->memory = appMemory ? appMemory : (const void *)new char[bytes_for(type, n1, n2, n3)];
  o->deleter = appMemory ? deleter : owned_memory_deleter;
  o->userdata = appMemory ? userData : nullptr;
  return (ANARIArray3D)h;
}

void *PhotonDevice::mapArray(ANARIArray a)
{
  auto *o = get((ANARIObject)a);
  return o ? const_cast<void *>(o->memory) : nullptr;
}

void PhotonDevice::unmapArray(ANARIArray) {}

ANARIGeometry PhotonDevice::newGeometry(const char *type)
{
  const uintptr_t h = alloc_handle(ANARI_GEOMETRY);
  auto *o = get((ANARIObject)h);
  if (o && type) {
    const size_t n = std::strlen(type) + 1;
    o->params["subtype"] = std::vector<std::byte>(n);
    std::memcpy(o->params["subtype"].data(), type, n);
  }
  return (ANARIGeometry)h;
}
ANARISurface PhotonDevice::newSurface() { return (ANARISurface)alloc_handle(ANARI_SURFACE); }
ANARIWorld PhotonDevice::newWorld() { return (ANARIWorld)alloc_handle(ANARI_WORLD); }
ANARIRenderer PhotonDevice::newRenderer(const char *) { return (ANARIRenderer)alloc_handle(ANARI_RENDERER); }
ANARIFrame PhotonDevice::newFrame() { return (ANARIFrame)alloc_handle(ANARI_FRAME); }

ANARILight PhotonDevice::newLight(const char *) { return nullptr; }
ANARICamera PhotonDevice::newCamera(const char *) { return nullptr; }
ANARISpatialField PhotonDevice::newSpatialField(const char *) { return nullptr; }
ANARIVolume PhotonDevice::newVolume(const char *) { return nullptr; }
ANARIMaterial PhotonDevice::newMaterial(const char *) { return nullptr; }
ANARISampler PhotonDevice::newSampler(const char *) { return nullptr; }
ANARIGroup PhotonDevice::newGroup() { return nullptr; }
ANARIInstance PhotonDevice::newInstance(const char *) { return nullptr; }

const char **PhotonDevice::getObjectSubtypes(ANARIDataType)
{
  static const char *empty[] = {nullptr};
  return empty;
}

const void *PhotonDevice::getObjectInfo(ANARIDataType, const char *, const char *, ANARIDataType) { return nullptr; }

const void *PhotonDevice::getParameterInfo(
    ANARIDataType, const char *, const char *, ANARIDataType, const char *, ANARIDataType)
{
  return nullptr;
}

int PhotonDevice::getProperty(ANARIObject, const char *name, ANARIDataType type, void *mem, uint64_t size, ANARIWaitMask)
{
  if (name && std::strcmp(name, "version") == 0 && type == ANARI_INT32 && size >= sizeof(int)) {
    *(int *)mem = 1;
    return 1;
  }
  return 0;
}

void PhotonDevice::setParameter(ANARIObject object, const char *name, ANARIDataType type, const void *mem)
{
  auto *o = get(object);
  if (!o || !name || !mem)
    return;

  if (type == ANARI_ARRAY1D || type == ANARI_ARRAY2D || type == ANARI_ARRAY3D || type == ANARI_GEOMETRY
      || type == ANARI_SURFACE || type == ANARI_WORLD || type == ANARI_FRAME || type == ANARI_RENDERER
      || type == ANARI_CAMERA || type == ANARI_LIGHT || type == ANARI_MATERIAL) {
    uintptr_t h = 0;
    std::memcpy(&h, mem, sizeof(uintptr_t));
    std::vector<std::byte> bytes(sizeof(uintptr_t));
    std::memcpy(bytes.data(), &h, sizeof(uintptr_t));
    o->params[name] = std::move(bytes);
    return;
  }

  const size_t n = sizeof_anari(type);
  std::vector<std::byte> bytes(n);
  std::memcpy(bytes.data(), mem, n);
  o->params[name] = std::move(bytes);
}

void PhotonDevice::unsetParameter(ANARIObject object, const char *name)
{
  auto *o = get(object);
  if (!o || !name)
    return;
  o->params.erase(name);
}

void PhotonDevice::unsetAllParameters(ANARIObject object)
{
  auto *o = get(object);
  if (!o)
    return;
  o->params.clear();
}

void PhotonDevice::commitParameters(ANARIObject) {}

void PhotonDevice::release(ANARIObject object)
{
  auto it = m_objects.find((uintptr_t)object);
  if (it == m_objects.end())
    return;

  it->second->refcount -= 1;
  if (it->second->refcount > 0)
    return;

  if (it->second->deleter)
    it->second->deleter(it->second->userdata, it->second->memory);

  m_objects.erase(it);
}

void PhotonDevice::retain(ANARIObject object)
{
  auto *o = get(object);
  if (o)
    o->refcount += 1;
}

void *PhotonDevice::mapParameterArray1D(ANARIObject, const char *, ANARIDataType, uint64_t, uint64_t *)
{
  return nullptr;
}

void *PhotonDevice::mapParameterArray2D(ANARIObject, const char *, ANARIDataType, uint64_t, uint64_t, uint64_t *)
{
  return nullptr;
}

void *PhotonDevice::mapParameterArray3D(
    ANARIObject, const char *, ANARIDataType, uint64_t, uint64_t, uint64_t, uint64_t *)
{
  return nullptr;
}

void PhotonDevice::unmapParameterArray(ANARIObject, const char *) {}

const void *PhotonDevice::frameBufferMap(ANARIFrame fb, const char *, uint32_t *w, uint32_t *h, ANARIDataType *t)
{
  auto *o = get((ANARIObject)fb);
  if (!o) {
    if (w)
      *w = 0;
    if (h)
      *h = 0;
    if (t)
      *t = ANARI_UNKNOWN;
    return nullptr;
  }

  uint32_t size[2] = {512u, 512u};
  auto it = o->params.find("size");
  if (it != o->params.end() && it->second.size() == sizeof(size))
    std::memcpy(size, it->second.data(), sizeof(size));

  m_fb_w = size[0];
  m_fb_h = size[1];
  m_fb_bytes.resize(size_t(m_fb_w) * size_t(m_fb_h) * 4 * sizeof(float));

  if (w)
    *w = m_fb_w;
  if (h)
    *h = m_fb_h;
  if (t)
    *t = ANARI_FLOAT32_VEC4;
  return m_fb_bytes.data();
}

void PhotonDevice::frameBufferUnmap(ANARIFrame, const char *) {}

void PhotonDevice::renderFrame(ANARIFrame fb)
{
  auto *o = get((ANARIObject)fb);
  if (!o)
    return;

  uint32_t size[2] = {512u, 512u};
  auto it = o->params.find("size");
  if (it != o->params.end() && it->second.size() == sizeof(size))
    std::memcpy(size, it->second.data(), sizeof(size));

  m_fb_w = size[0];
  m_fb_h = size[1];
  m_fb_bytes.resize(size_t(m_fb_w) * size_t(m_fb_h) * 4 * sizeof(float));

  auto *out = reinterpret_cast<float *>(m_fb_bytes.data());

  photon::pt::PathTracer pt;
  pt.params.width = m_fb_w;
  pt.params.height = m_fb_h;
  pt.params.samples_per_pixel = 16;
  pt.params.max_depth = 4;

  std::optional<photon::pt::Scene> scene;

  uintptr_t world_h = 0;
  auto wit = o->params.find("world");
  if (wit != o->params.end() && wit->second.size() == sizeof(uintptr_t))
    std::memcpy(&world_h, wit->second.data(), sizeof(uintptr_t));

  if (world_h != 0) {
    auto *wo = get((ANARIObject)world_h);
    if (wo) {
      uintptr_t surfaces_arr = 0;
      auto sit = wo->params.find("surface");
      if (sit != wo->params.end() && sit->second.size() == sizeof(uintptr_t))
        std::memcpy(&surfaces_arr, sit->second.data(), sizeof(uintptr_t));

      auto *sa = get((ANARIObject)surfaces_arr);
      if (sa && sa->object_type == ANARI_ARRAY1D && sa->memory) {
        const auto *surfaces = reinterpret_cast<const uintptr_t *>(sa->memory);

        const uintptr_t surf_h = surfaces[0];
        auto *so = get((ANARIObject)surf_h);
        if (so) {
          uintptr_t geom_h = 0;
          auto git = so->params.find("geometry");
          if (git != so->params.end() && git->second.size() == sizeof(uintptr_t))
            std::memcpy(&geom_h, git->second.data(), sizeof(uintptr_t));

          auto *go = get((ANARIObject)geom_h);
          if (go) {
            uintptr_t vtx_h = 0;
            auto vit = go->params.find("vertex.position");
            if (vit != go->params.end() && vit->second.size() == sizeof(uintptr_t))
              std::memcpy(&vtx_h, vit->second.data(), sizeof(uintptr_t));

            auto *va = get((ANARIObject)vtx_h);
            if (va && va->object_type == ANARI_ARRAY1D && va->memory) {
              photon::pt::TriangleMesh mesh;
              mesh.positions = Kokkos::View<photon::pt::Vec3 *>("pos", 3);
              mesh.indices = Kokkos::View<photon::pt::u32 *>("idx", 3);
              mesh.albedo_per_prim = Kokkos::View<photon::pt::Vec3 *>("alb", 1);


              auto pos_h = Kokkos::create_mirror_view(mesh.positions);
              auto idx_h = Kokkos::create_mirror_view(mesh.indices);
              auto alb_h = Kokkos::create_mirror_view(mesh.albedo_per_prim);

              const float *v = reinterpret_cast<const float *>(va->memory);
              pos_h(0) = {v[0], v[1], v[2]};
              pos_h(1) = {v[3], v[4], v[5]};
              pos_h(2) = {v[6], v[7], v[8]};

              idx_h(0) = 0;
              idx_h(1) = 1;
              idx_h(2) = 2;
              alb_h(0) = {1.0f, 0.0f, 0.0f};

              Kokkos::deep_copy(mesh.positions, pos_h);
              Kokkos::deep_copy(mesh.indices, idx_h);
              Kokkos::deep_copy(mesh.albedo_per_prim, alb_h);

              photon::pt::Scene s;
              s.mesh = mesh;
              s.bvh = photon::pt::Bvh::build_cpu(mesh);
              scene = std::move(s);
            }
          }
        }
      }
    }
  }

  if (scene)
    pt.scene = std::move(*scene);
  else
    pt.scene = photon::pt::SceneBuilder::make_two_quads();


  auto pixels = pt.render();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pixels);

  for (uint32_t y = 0; y < m_fb_h; ++y) {
    for (uint32_t x = 0; x < m_fb_w; ++x) {
      const size_t idx = size_t(y) * size_t(m_fb_w) + x;
      auto c = photon::pt::clamp01(host(y, x));
      out[4 * idx + 0] = c.x;
      out[4 * idx + 1] = c.y;
      out[4 * idx + 2] = c.z;
      out[4 * idx + 3] = 1.f;
    }
  }
}

int PhotonDevice::frameReady(ANARIFrame, ANARIWaitMask) { return 1; }

void PhotonDevice::discardFrame(ANARIFrame) {}

}
