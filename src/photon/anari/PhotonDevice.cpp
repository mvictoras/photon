#include <anari/anari.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "photon/anari/PhotonDevice.h"

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

ANARIGeometry PhotonDevice::newGeometry(const char *) { return (ANARIGeometry)alloc_handle(ANARI_GEOMETRY); }
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
  m_framebuffer.assign(size_t(m_fb_w) * size_t(m_fb_h) * 4, 0.f);

  for (uint32_t y = 0; y < m_fb_h; ++y) {
    for (uint32_t x = 0; x < m_fb_w; ++x) {
      const size_t idx = size_t(y) * size_t(m_fb_w) + x;
      m_framebuffer[4 * idx + 0] = float(x) / float(m_fb_w - 1);
      m_framebuffer[4 * idx + 1] = float(y) / float(m_fb_h - 1);
      m_framebuffer[4 * idx + 2] = 0.2f;
      m_framebuffer[4 * idx + 3] = 1.f;
    }
  }
}

int PhotonDevice::frameReady(ANARIFrame, ANARIWaitMask) { return 1; }

void PhotonDevice::discardFrame(ANARIFrame) {}

}
