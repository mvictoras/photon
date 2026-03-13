#include <anari/anari.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#include "photon/pt/pathtracer.h"
#include "photon/pt/backend/kokkos_backend.h"
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
  case ANARI_UINT32:
  case ANARI_FLOAT32:
    return 4;
  case ANARI_FLOAT32_VEC2:
  case ANARI_UINT32_VEC2:
    return 8;
  case ANARI_FLOAT32_VEC3:
  case ANARI_UINT32_VEC3:
    return 12;
  case ANARI_FLOAT32_VEC4:
  case ANARI_UINT32_VEC4:
    return 16;
  case ANARI_FLOAT32_MAT4:
    return 64;
  default:
    break;
  }

  if (type == ANARI_SURFACE || type == ANARI_GEOMETRY || type == ANARI_ARRAY1D
      || type == ANARI_ARRAY2D || type == ANARI_ARRAY3D || type == ANARI_WORLD
      || type == ANARI_FRAME || type == ANARI_RENDERER || type == ANARI_CAMERA
      || type == ANARI_LIGHT || type == ANARI_MATERIAL || type == ANARI_GROUP
      || type == ANARI_INSTANCE || type == ANARI_SAMPLER || type == ANARI_VOLUME
      || type == ANARI_SPATIAL_FIELD || type == ANARI_DEVICE)
    return sizeof(uintptr_t);

  return 0;
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

static bool is_handle_type(ANARIDataType type)
{
  return type == ANARI_SURFACE || type == ANARI_GEOMETRY || type == ANARI_ARRAY1D
      || type == ANARI_ARRAY2D || type == ANARI_ARRAY3D || type == ANARI_WORLD
      || type == ANARI_FRAME || type == ANARI_RENDERER || type == ANARI_CAMERA
      || type == ANARI_LIGHT || type == ANARI_MATERIAL || type == ANARI_GROUP
      || type == ANARI_INSTANCE || type == ANARI_SAMPLER || type == ANARI_VOLUME
      || type == ANARI_SPATIAL_FIELD || type == ANARI_DEVICE;
}

ANARIArray1D PhotonDevice::newArray1D(
    const void *appMemory, ANARIMemoryDeleter, const void *, ANARIDataType type, uint64_t n1)
{
  const uintptr_t h = alloc_handle(ANARI_ARRAY1D);
  auto *o = get((ANARIObject)h);
  o->array_element_type = type;
  o->array_num_items1 = n1;

  const size_t nb = bytes_for(type, n1);
  auto *buf = new char[nb > 0 ? nb : 1];
  if (appMemory && nb > 0)
    std::memcpy(buf, appMemory, nb);
  o->memory = buf;
  o->deleter = owned_memory_deleter;
  o->userdata = nullptr;

  if (is_handle_type(type) && appMemory) {
    const auto *handles = reinterpret_cast<const uintptr_t *>(appMemory);
    for (uint64_t i = 0; i < n1; ++i) {
      if (handles[i] != 0)
        retain((ANARIObject)handles[i]);
    }
  }

  return (ANARIArray1D)h;
}

ANARIArray2D PhotonDevice::newArray2D(
    const void *appMemory, ANARIMemoryDeleter, const void *, ANARIDataType type, uint64_t n1, uint64_t n2)
{
  const uintptr_t h = alloc_handle(ANARI_ARRAY2D);
  auto *o = get((ANARIObject)h);
  o->array_element_type = type;
  o->array_num_items1 = n1;
  o->array_num_items2 = n2;

  const size_t nb = bytes_for(type, n1, n2);
  auto *buf = new char[nb > 0 ? nb : 1];
  if (appMemory && nb > 0)
    std::memcpy(buf, appMemory, nb);
  o->memory = buf;
  o->deleter = owned_memory_deleter;
  o->userdata = nullptr;
  return (ANARIArray2D)h;
}

ANARIArray3D PhotonDevice::newArray3D(const void *appMemory, ANARIMemoryDeleter, const void *,
    ANARIDataType type, uint64_t n1, uint64_t n2, uint64_t n3)
{
  const uintptr_t h = alloc_handle(ANARI_ARRAY3D);
  auto *o = get((ANARIObject)h);
  o->array_element_type = type;
  o->array_num_items1 = n1;
  o->array_num_items2 = n2;
  o->array_num_items3 = n3;

  const size_t nb = bytes_for(type, n1, n2, n3);
  auto *buf = new char[nb > 0 ? nb : 1];
  if (appMemory && nb > 0)
    std::memcpy(buf, appMemory, nb);
  o->memory = buf;
  o->deleter = owned_memory_deleter;
  o->userdata = nullptr;
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

ANARILight PhotonDevice::newLight(const char *type)
{
  const uintptr_t h = alloc_handle(ANARI_LIGHT);
  auto *o = get((ANARIObject)h);
  if (o && type) {
    const size_t n = std::strlen(type) + 1;
    o->params["subtype"] = std::vector<std::byte>(n);
    std::memcpy(o->params["subtype"].data(), type, n);
  }
  return (ANARILight)h;
}

ANARICamera PhotonDevice::newCamera(const char *type)
{
  const uintptr_t h = alloc_handle(ANARI_CAMERA);
  auto *o = get((ANARIObject)h);
  if (o && type) {
    const size_t n = std::strlen(type) + 1;
    o->params["subtype"] = std::vector<std::byte>(n);
    std::memcpy(o->params["subtype"].data(), type, n);
  }
  return (ANARICamera)h;
}
ANARISpatialField PhotonDevice::newSpatialField(const char *) { return nullptr; }
ANARIVolume PhotonDevice::newVolume(const char *) { return nullptr; }
ANARIMaterial PhotonDevice::newMaterial(const char *type)
{
  const uintptr_t h = alloc_handle(ANARI_MATERIAL);
  auto *o = get((ANARIObject)h);
  if (o && type) {
    const size_t n = std::strlen(type) + 1;
    o->params["subtype"] = std::vector<std::byte>(n);
    std::memcpy(o->params["subtype"].data(), type, n);
  }
  return (ANARIMaterial)h;
}

ANARISampler PhotonDevice::newSampler(const char *) { return nullptr; }

ANARIGroup PhotonDevice::newGroup() { return (ANARIGroup)alloc_handle(ANARI_GROUP); }

ANARIInstance PhotonDevice::newInstance(const char *type)
{
  const uintptr_t h = alloc_handle(ANARI_INSTANCE);
  auto *o = get((ANARIObject)h);
  if (o && type) {
    const size_t n = std::strlen(type) + 1;
    o->params["subtype"] = std::vector<std::byte>(n);
    std::memcpy(o->params["subtype"].data(), type, n);
  }
  return (ANARIInstance)h;
}

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

int PhotonDevice::getProperty(ANARIObject object, const char *name, ANARIDataType type, void *mem, uint64_t size, ANARIWaitMask)
{
  if (name && std::strcmp(name, "version") == 0 && type == ANARI_INT32 && size >= sizeof(int)) {
    *(int *)mem = 1;
    return 1;
  }

  if (name && std::strcmp(name, "bounds") == 0 && size >= 6 * sizeof(float)) {
    auto *wo = get(object);
    if (!wo || wo->object_type != ANARI_WORLD)
      return 0;

    float lo[3] = {1e30f, 1e30f, 1e30f};
    float hi[3] = {-1e30f, -1e30f, -1e30f};
    bool found = false;

    auto expand_vertex = [&](const float *p) {
      for (int c = 0; c < 3; ++c) {
        if (p[c] < lo[c]) lo[c] = p[c];
        if (p[c] > hi[c]) hi[c] = p[c];
      }
      found = true;
    };

    auto scan_surface_positions = [&](uintptr_t surf_h) {
      auto *so = get((ANARIObject)surf_h);
      if (!so || so->object_type != ANARI_SURFACE)
        return;
      uintptr_t geom_h = 0;
      auto git = so->params.find("geometry");
      if (git != so->params.end() && git->second.size() == sizeof(uintptr_t))
        std::memcpy(&geom_h, git->second.data(), sizeof(uintptr_t));
      if (geom_h == 0)
        return;
      auto *go = get((ANARIObject)geom_h);
      if (!go || go->object_type != ANARI_GEOMETRY)
        return;
      uintptr_t pos_h = 0;
      auto pit = go->params.find("vertex.position");
      if (pit != go->params.end() && pit->second.size() == sizeof(uintptr_t))
        std::memcpy(&pos_h, pit->second.data(), sizeof(uintptr_t));
      if (pos_h == 0)
        return;
      auto *pa = get((ANARIObject)pos_h);
      if (!pa || pa->object_type != ANARI_ARRAY1D || !pa->memory || pa->array_element_type != ANARI_FLOAT32_VEC3)
        return;
      const float *verts = reinterpret_cast<const float *>(pa->memory);
      for (uint64_t i = 0; i < pa->array_num_items1; ++i)
        expand_vertex(verts + 3 * i);
    };

    auto scan_array = [&](const char *param_name, ANARIDataType /*elem_type*/, auto handler) {
      uintptr_t arr_h = 0;
      auto ait = wo->params.find(param_name);
      if (ait == wo->params.end() || ait->second.size() != sizeof(uintptr_t))
        return;
      std::memcpy(&arr_h, ait->second.data(), sizeof(uintptr_t));
      if (arr_h == 0)
        return;
      auto *arr = get((ANARIObject)arr_h);
      if (!arr || arr->object_type != ANARI_ARRAY1D || !arr->memory)
        return;
      const auto *handles = reinterpret_cast<const uintptr_t *>(arr->memory);
      for (uint64_t i = 0; i < arr->array_num_items1; ++i)
        handler(handles[i]);
    };

    scan_array("surface", ANARI_SURFACE, scan_surface_positions);

    scan_array("instance", ANARI_INSTANCE, [&](uintptr_t inst_h) {
      auto *inst = get((ANARIObject)inst_h);
      if (!inst || inst->object_type != ANARI_INSTANCE)
        return;
      uintptr_t group_h = 0;
      auto git = inst->params.find("group");
      if (git != inst->params.end() && git->second.size() == sizeof(uintptr_t))
        std::memcpy(&group_h, git->second.data(), sizeof(uintptr_t));
      if (group_h == 0)
        return;
      auto *grp = get((ANARIObject)group_h);
      if (!grp || grp->object_type != ANARI_GROUP)
        return;
      uintptr_t gsarr_h = 0;
      auto sit = grp->params.find("surface");
      if (sit != grp->params.end() && sit->second.size() == sizeof(uintptr_t))
        std::memcpy(&gsarr_h, sit->second.data(), sizeof(uintptr_t));
      if (gsarr_h == 0)
        return;
      auto *gsa = get((ANARIObject)gsarr_h);
      if (!gsa || gsa->object_type != ANARI_ARRAY1D || !gsa->memory)
        return;

      float xfm_raw[16];
      bool has_xfm = false;
      auto xit = inst->params.find("transform");
      if (xit != inst->params.end() && xit->second.size() == 16 * sizeof(float)) {
        std::memcpy(xfm_raw, xit->second.data(), sizeof(xfm_raw));
        has_xfm = true;
      }

      const auto *shandles = reinterpret_cast<const uintptr_t *>(gsa->memory);
      for (uint64_t si = 0; si < gsa->array_num_items1; ++si) {
        auto *sso = get((ANARIObject)shandles[si]);
        if (!sso || sso->object_type != ANARI_SURFACE)
          continue;
        uintptr_t gg = 0;
        auto ggi = sso->params.find("geometry");
        if (ggi != sso->params.end() && ggi->second.size() == sizeof(uintptr_t))
          std::memcpy(&gg, ggi->second.data(), sizeof(uintptr_t));
        if (gg == 0) continue;
        auto *ggo = get((ANARIObject)gg);
        if (!ggo || ggo->object_type != ANARI_GEOMETRY) continue;
        uintptr_t pp = 0;
        auto ppi = ggo->params.find("vertex.position");
        if (ppi != ggo->params.end() && ppi->second.size() == sizeof(uintptr_t))
          std::memcpy(&pp, ppi->second.data(), sizeof(uintptr_t));
        if (pp == 0) continue;
        auto *ppa = get((ANARIObject)pp);
        if (!ppa || ppa->object_type != ANARI_ARRAY1D || !ppa->memory || ppa->array_element_type != ANARI_FLOAT32_VEC3) continue;
        const float *verts = reinterpret_cast<const float *>(ppa->memory);
        for (uint64_t vi = 0; vi < ppa->array_num_items1; ++vi) {
          float p[3] = {verts[3 * vi], verts[3 * vi + 1], verts[3 * vi + 2]};
          if (has_xfm) {
            float x = xfm_raw[0] * p[0] + xfm_raw[4] * p[1] + xfm_raw[8] * p[2] + xfm_raw[12];
            float y = xfm_raw[1] * p[0] + xfm_raw[5] * p[1] + xfm_raw[9] * p[2] + xfm_raw[13];
            float z = xfm_raw[2] * p[0] + xfm_raw[6] * p[1] + xfm_raw[10] * p[2] + xfm_raw[14];
            p[0] = x; p[1] = y; p[2] = z;
          }
          expand_vertex(p);
        }
      }
    });

    if (!found)
      return 0;

    float *out = reinterpret_cast<float *>(mem);
    out[0] = lo[0]; out[1] = lo[1]; out[2] = lo[2];
    out[3] = hi[0]; out[4] = hi[1]; out[5] = hi[2];
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
      || type == ANARI_CAMERA || type == ANARI_LIGHT || type == ANARI_MATERIAL
      || type == ANARI_GROUP || type == ANARI_INSTANCE || type == ANARI_SAMPLER
      || type == ANARI_SPATIAL_FIELD || type == ANARI_VOLUME) {
    uintptr_t h = 0;
    std::memcpy(&h, mem, sizeof(uintptr_t));

    if (h != 0)
      retain((ANARIObject)h);

    auto prev = o->params.find(name);
    if (prev != o->params.end() && prev->second.size() == sizeof(uintptr_t)) {
      uintptr_t old_h = 0;
      std::memcpy(&old_h, prev->second.data(), sizeof(uintptr_t));
      if (old_h != 0)
        release((ANARIObject)old_h);
    }

    std::vector<std::byte> bytes(sizeof(uintptr_t));
    std::memcpy(bytes.data(), &h, sizeof(uintptr_t));
    o->params[name] = std::move(bytes);
    return;
  }

  if (type == ANARI_STRING) {
    const char *s = reinterpret_cast<const char *>(mem);
    if (!s)
      return;

    const size_t n = std::strlen(s) + 1;
    std::vector<std::byte> bytes(n);
    std::memcpy(bytes.data(), s, n);
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

void *PhotonDevice::mapParameterArray1D(
    ANARIObject object, const char *name, ANARIDataType type, uint64_t n1, uint64_t *stride)
{
  auto arr = newArray1D(nullptr, nullptr, nullptr, type, n1);
  setParameter(object, name, ANARI_ARRAY1D, &arr);
  if (stride)
    *stride = sizeof_anari(type);
  release(arr);
  return mapArray(arr);
}

void *PhotonDevice::mapParameterArray2D(
    ANARIObject object, const char *name, ANARIDataType type, uint64_t n1, uint64_t n2, uint64_t *stride)
{
  auto arr = newArray2D(nullptr, nullptr, nullptr, type, n1, n2);
  setParameter(object, name, ANARI_ARRAY2D, &arr);
  if (stride)
    *stride = sizeof_anari(type);
  release(arr);
  return mapArray(arr);
}

void *PhotonDevice::mapParameterArray3D(
    ANARIObject object, const char *name, ANARIDataType type, uint64_t n1, uint64_t n2, uint64_t n3, uint64_t *stride)
{
  auto arr = newArray3D(nullptr, nullptr, nullptr, type, n1, n2, n3);
  setParameter(object, name, ANARI_ARRAY3D, &arr);
  if (stride)
    *stride = sizeof_anari(type);
  release(arr);
  return mapArray(arr);
}

void PhotonDevice::unmapParameterArray(ANARIObject object, const char *name)
{
  auto *o = get(object);
  if (!o || !name)
    return;

  auto it = o->params.find(name);
  if (it == o->params.end() || it->second.size() != sizeof(uintptr_t))
    return;

  uintptr_t arr_h = 0;
  std::memcpy(&arr_h, it->second.data(), sizeof(uintptr_t));
  auto *arr = get((ANARIObject)arr_h);
  if (!arr || !arr->memory)
    return;

  if (is_handle_type(arr->array_element_type)) {
    const auto *handles = reinterpret_cast<const uintptr_t *>(arr->memory);
    for (uint64_t i = 0; i < arr->array_num_items1; ++i) {
      if (handles[i] != 0)
        retain((ANARIObject)handles[i]);
    }
  }
}

const void *PhotonDevice::frameBufferMap(ANARIFrame fb, const char *channel, uint32_t *w, uint32_t *h, ANARIDataType *t)
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

  if (w)
    *w = m_fb_w;
  if (h)
    *h = m_fb_h;

  const char *chan = channel ? channel : "channel.color";

  if (std::strcmp(chan, "color") == 0 || std::strcmp(chan, "channel.color") == 0) {
    if (t)
      *t = ANARI_FLOAT32_VEC4;
    m_fb_bytes.resize(size_t(m_fb_w) * size_t(m_fb_h) * 4 * sizeof(float));
    return m_fb_bytes.data();
  }

  if (std::strcmp(chan, "depth") == 0) {
    if (t)
      *t = ANARI_FLOAT32;
    return m_channel_depth.data();
  }

  if (std::strcmp(chan, "normal") == 0) {
    if (t)
      *t = ANARI_FLOAT32_VEC3;
    return m_channel_normal.data();
  }

  if (std::strcmp(chan, "albedo") == 0) {
    if (t)
      *t = ANARI_FLOAT32_VEC3;
    return m_channel_albedo.data();
  }

  if (t)
    *t = ANARI_UNKNOWN;
  return nullptr;
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

  uintptr_t world_h = 0;
  auto wit = o->params.find("world");
  if (wit != o->params.end() && wit->second.size() == sizeof(uintptr_t))
    std::memcpy(&world_h, wit->second.data(), sizeof(uintptr_t));

  uintptr_t renderer_h = 0;
  auto rit = o->params.find("renderer");
  if (rit != o->params.end() && rit->second.size() == sizeof(uintptr_t))
    std::memcpy(&renderer_h, rit->second.data(), sizeof(uintptr_t));

  uint32_t spp = 16;
  uint32_t max_depth = 5;

  if (renderer_h != 0) {
    auto *ro = get((ANARIObject)renderer_h);
    if (ro) {
      auto pit = ro->params.find("samples_per_pixel");
      if (pit != ro->params.end() && pit->second.size() == sizeof(uint32_t))
        std::memcpy(&spp, pit->second.data(), sizeof(uint32_t));
      pit = ro->params.find("spp");
      if (pit != ro->params.end() && pit->second.size() == sizeof(uint32_t))
        std::memcpy(&spp, pit->second.data(), sizeof(uint32_t));
      pit = ro->params.find("pixelSamples");
      if (pit != ro->params.end() && pit->second.size() == sizeof(uint32_t))
        std::memcpy(&spp, pit->second.data(), sizeof(uint32_t));

      pit = ro->params.find("max_depth");
      if (pit != ro->params.end() && pit->second.size() == sizeof(uint32_t))
        std::memcpy(&max_depth, pit->second.data(), sizeof(uint32_t));
      pit = ro->params.find("maxDepth");
      if (pit != ro->params.end() && pit->second.size() == sizeof(uint32_t))
        std::memcpy(&max_depth, pit->second.data(), sizeof(uint32_t));
    }
  }

  auto scene_opt = build_scene_from_anari((ANARIWorld)world_h, *this);
  photon::pt::Scene scene = scene_opt ? *scene_opt : photon::pt::SceneBuilder::make_two_quads();

  photon::pt::Camera cam = build_camera_from_anari(fb, *this);

  auto backend = std::make_unique<photon::pt::KokkosBackend>();
  backend->build_accel(scene);

  photon::pt::PathTracer pt;
  pt.params.width = m_fb_w;
  pt.params.height = m_fb_h;
  pt.params.samples_per_pixel = spp;
  pt.params.max_depth = max_depth;
  pt.camera = cam;
  pt.set_scene(scene);
  pt.set_backend(std::move(backend));

  auto result = pt.render();

  auto color_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
  auto depth_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.depth);
  auto normal_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.normal);
  auto albedo_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.albedo);

  m_channel_color.resize(size_t(m_fb_w) * size_t(m_fb_h) * 4);
  m_channel_depth.resize(size_t(m_fb_w) * size_t(m_fb_h));
  m_channel_normal.resize(size_t(m_fb_w) * size_t(m_fb_h) * 3);
  m_channel_albedo.resize(size_t(m_fb_w) * size_t(m_fb_h) * 3);

  for (uint32_t y = 0; y < m_fb_h; ++y) {
    for (uint32_t x = 0; x < m_fb_w; ++x) {
      const size_t idx = size_t(y) * size_t(m_fb_w) + x;

      const auto c = photon::pt::clamp01(color_host(y, x));
      out[4 * idx + 0] = c.x;
      out[4 * idx + 1] = c.y;
      out[4 * idx + 2] = c.z;
      out[4 * idx + 3] = 1.f;

      m_channel_color[4 * idx + 0] = c.x;
      m_channel_color[4 * idx + 1] = c.y;
      m_channel_color[4 * idx + 2] = c.z;
      m_channel_color[4 * idx + 3] = 1.f;

      m_channel_depth[idx] = depth_host(y, x);

      const auto n = normal_host(y, x);
      m_channel_normal[3 * idx + 0] = n.x;
      m_channel_normal[3 * idx + 1] = n.y;
      m_channel_normal[3 * idx + 2] = n.z;

      const auto a = albedo_host(y, x);
      m_channel_albedo[3 * idx + 0] = a.x;
      m_channel_albedo[3 * idx + 1] = a.y;
      m_channel_albedo[3 * idx + 2] = a.z;
    }
  }
}

int PhotonDevice::frameReady(ANARIFrame, ANARIWaitMask) { return 1; }

void PhotonDevice::report(ANARIObject, ANARIDataType, ANARIStatusSeverity, ANARIStatusCode, const char *) {}

void PhotonDevice::discardFrame(ANARIFrame) {}

}
