#include <anari/anari.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>

#include "photon/anari/PhotonDevice.h"

#include <Kokkos_Core.hpp>

#include "photon/pt/pathtracer.h"
// denoiser.h included via PhotonDevice.h
#include "photon/pt/backend/ray_backend.h"
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
  // 1-byte element types
  case ANARI_UINT8:
  case ANARI_INT8:
  case ANARI_UFIXED8:
  case ANARI_FIXED8:
  case ANARI_UFIXED8_R_SRGB:
    return 1;
  case ANARI_UINT8_VEC2:
  case ANARI_INT8_VEC2:
  case ANARI_UFIXED8_VEC2:
  case ANARI_FIXED8_VEC2:
  case ANARI_UFIXED8_RA_SRGB:
    return 2;
  case ANARI_UINT8_VEC3:
  case ANARI_INT8_VEC3:
  case ANARI_UFIXED8_VEC3:
  case ANARI_FIXED8_VEC3:
  case ANARI_UFIXED8_RGB_SRGB:
    return 3;
  case ANARI_UINT8_VEC4:
  case ANARI_INT8_VEC4:
  case ANARI_UFIXED8_VEC4:
  case ANARI_FIXED8_VEC4:
  case ANARI_UFIXED8_RGBA_SRGB:
    return 4;
  // 2-byte element types
  case ANARI_UINT16:
  case ANARI_INT16:
  case ANARI_UFIXED16:
  case ANARI_FIXED16:
    return 2;
  case ANARI_UINT16_VEC2:
  case ANARI_INT16_VEC2:
  case ANARI_UFIXED16_VEC2:
  case ANARI_FIXED16_VEC2:
    return 4;
  case ANARI_UINT16_VEC3:
  case ANARI_INT16_VEC3:
  case ANARI_UFIXED16_VEC3:
  case ANARI_FIXED16_VEC3:
    return 6;
  case ANARI_UINT16_VEC4:
  case ANARI_INT16_VEC4:
  case ANARI_UFIXED16_VEC4:
  case ANARI_FIXED16_VEC4:
    return 8;
  // 4-byte element types
  case ANARI_INT32:
  case ANARI_UINT32:
  case ANARI_FLOAT32:
    return 4;
  case ANARI_FLOAT32_VEC2:
  case ANARI_UINT32_VEC2:
  case ANARI_INT32_VEC2:
    return 8;
  case ANARI_FLOAT32_VEC3:
  case ANARI_UINT32_VEC3:
  case ANARI_INT32_VEC3:
    return 12;
  case ANARI_FLOAT32_VEC4:
  case ANARI_UINT32_VEC4:
  case ANARI_INT32_VEC4:
    return 16;
  // 8-byte element types
  case ANARI_FLOAT64:
  case ANARI_INT64:
  case ANARI_UINT64:
    return 8;
  case ANARI_FLOAT64_VEC2:
  case ANARI_INT64_VEC2:
  case ANARI_UINT64_VEC2:
    return 16;
  case ANARI_FLOAT64_VEC3:
  case ANARI_INT64_VEC3:
  case ANARI_UINT64_VEC3:
    return 24;
  case ANARI_FLOAT64_VEC4:
  case ANARI_INT64_VEC4:
  case ANARI_UINT64_VEC4:
    return 32;
  // Matrix types
  case ANARI_FLOAT32_MAT4:
    return 64;
  // Bool
  case ANARI_BOOL:
    return 4;
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

PhotonDevice::PhotonDevice(ANARILibrary)
{
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
}

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

ANARISampler PhotonDevice::newSampler(const char *type)
{
  const uintptr_t h = alloc_handle(ANARI_SAMPLER);
  auto *o = get((ANARIObject)h);
  if (o && type) {
    const size_t n = std::strlen(type) + 1;
    o->params["subtype"] = std::vector<std::byte>(n);
    std::memcpy(o->params["subtype"].data(), type, n);
  }
  return (ANARISampler)h;
}

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

const char **PhotonDevice::getObjectSubtypes(ANARIDataType objectType)
{
  static const char *renderer[] = {"default", nullptr};
  static const char *camera[] = {"perspective", nullptr};
  static const char *light[] = {"directional", "point", "quad", nullptr};
  static const char *geometry[] = {"triangle", "sphere", "cylinder", nullptr};
  static const char *material[] = {"matte", "physicallyBased", nullptr};
  static const char *sampler[] = {"image2D", nullptr};
  static const char *empty[] = {nullptr};

  switch (objectType) {
  case ANARI_RENDERER:
    return renderer;
  case ANARI_CAMERA:
    return camera;
  case ANARI_LIGHT:
    return light;
  case ANARI_GEOMETRY:
    return geometry;
  case ANARI_MATERIAL:
    return material;
  case ANARI_SAMPLER:
    return sampler;
  default:
    return empty;
  }
}

const void *PhotonDevice::getObjectInfo(
    ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  if (objectType == ANARI_DEVICE && infoName
      && std::strcmp(infoName, "extension") == 0
      && infoType == ANARI_STRING_LIST) {
    static const char *extensions[] = {nullptr};
    return extensions;
  }

  // Renderer "default" — report available parameters
  if (objectType == ANARI_RENDERER && infoName
      && std::strcmp(infoName, "parameter") == 0
      && infoType == ANARI_PARAMETER_LIST) {
    static const ANARIParameter params[] = {
        {"pixelSamples", ANARI_INT32},
        {"sampleLimit", ANARI_INT32},
        {"maxRayDepth", ANARI_INT32},
        {"background", ANARI_FLOAT32_VEC4},
        {"ambientColor", ANARI_FLOAT32_VEC3},
        {"ambientRadiance", ANARI_FLOAT32},
#ifdef PHOTON_HAS_OIDN
        {"denoise", ANARI_BOOL},
#endif
        {nullptr, ANARI_UNKNOWN},
    };
    return params;
  }

  return nullptr;
}

const void *PhotonDevice::getParameterInfo(
    ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  if (objectType != ANARI_RENDERER || !parameterName || !infoName)
    return nullptr;

  // --- pixelSamples (INT32) ---
  if (std::strcmp(parameterName, "pixelSamples") == 0
      && parameterType == ANARI_INT32) {
    if (std::strcmp(infoName, "default") == 0 && infoType == ANARI_INT32) {
      static const int32_t val = 1;
      return &val;
    }
    if (std::strcmp(infoName, "minimum") == 0 && infoType == ANARI_INT32) {
      static const int32_t val = 1;
      return &val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "samples per pixel per frame";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }

  // --- sampleLimit (INT32) ---
  if (std::strcmp(parameterName, "sampleLimit") == 0
      && parameterType == ANARI_INT32) {
    if (std::strcmp(infoName, "default") == 0 && infoType == ANARI_INT32) {
      static const int32_t val = 128;
      return &val;
    }
    if (std::strcmp(infoName, "minimum") == 0 && infoType == ANARI_INT32) {
      static const int32_t val = 0;
      return &val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "stop refining after this number of samples";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }

  // --- maxRayDepth (INT32) ---
  if (std::strcmp(parameterName, "maxRayDepth") == 0
      && parameterType == ANARI_INT32) {
    if (std::strcmp(infoName, "default") == 0 && infoType == ANARI_INT32) {
      static const int32_t val = 5;
      return &val;
    }
    if (std::strcmp(infoName, "minimum") == 0 && infoType == ANARI_INT32) {
      static const int32_t val = 1;
      return &val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "maximum number of ray bounces";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }

  // --- background (FLOAT32_VEC4) ---
  if (std::strcmp(parameterName, "background") == 0
      && parameterType == ANARI_FLOAT32_VEC4) {
    if (std::strcmp(infoName, "default") == 0
        && infoType == ANARI_FLOAT32_VEC4) {
      static const float val[4] = {0.f, 0.f, 0.f, 1.f};
      return val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "background color (RGBA)";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }

  // --- ambientColor (FLOAT32_VEC3) ---
  if (std::strcmp(parameterName, "ambientColor") == 0
      && parameterType == ANARI_FLOAT32_VEC3) {
    if (std::strcmp(infoName, "default") == 0
        && infoType == ANARI_FLOAT32_VEC3) {
      static const float val[3] = {1.f, 1.f, 1.f};
      return val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "color of the ambient light";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }

  // --- ambientRadiance (FLOAT32) ---
  if (std::strcmp(parameterName, "ambientRadiance") == 0
      && parameterType == ANARI_FLOAT32) {
    if (std::strcmp(infoName, "default") == 0 && infoType == ANARI_FLOAT32) {
      static const float val = 0.f;
      return &val;
    }
    if (std::strcmp(infoName, "minimum") == 0 && infoType == ANARI_FLOAT32) {
      static const float val = 0.f;
      return &val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "intensity of the ambient light";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }

#ifdef PHOTON_HAS_OIDN
  // --- denoise (BOOL) ---
  if (std::strcmp(parameterName, "denoise") == 0
      && parameterType == ANARI_BOOL) {
    if (std::strcmp(infoName, "default") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    if (std::strcmp(infoName, "description") == 0
        && infoType == ANARI_STRING) {
      static const char *desc = "enable denoising";
      return desc;
    }
    if (std::strcmp(infoName, "required") == 0 && infoType == ANARI_BOOL) {
      static const int32_t val = 0;
      return &val;
    }
    return nullptr;
  }
#endif

  return nullptr;
}

int PhotonDevice::getProperty(ANARIObject object, const char *name, ANARIDataType type, void *mem, uint64_t size, ANARIWaitMask)
{
  if (name && std::strcmp(name, "version") == 0 && type == ANARI_INT32 && size >= sizeof(int)) {
    *(int *)mem = 1;
    return 1;
  }

  if (name && std::strcmp(name, "numSamples") == 0 && type == ANARI_INT32 && size >= sizeof(int)) {
    *(int *)mem = m_frameID;
    return 1;
  }

  if (name && std::strcmp(name, "duration") == 0 && type == ANARI_FLOAT32 && size >= sizeof(float)) {
    *(float *)mem = m_lastDuration;
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

void PhotonDevice::commitParameters(ANARIObject object)
{
  auto *o = get(object);
  if (!o)
    return;

  // Bump world version when scene-affecting objects are committed
  switch (o->object_type) {
  case ANARI_WORLD:
  case ANARI_GEOMETRY:
  case ANARI_SURFACE:
  case ANARI_GROUP:
  case ANARI_INSTANCE:
  case ANARI_MATERIAL:
  case ANARI_LIGHT:
  case ANARI_SAMPLER:
    ++m_world_commit_version;
    break;
  case ANARI_RENDERER:
    ++m_renderer_version;
    break;
  case ANARI_CAMERA:
  case ANARI_FRAME:
    break;
  default:
    break;
  }
}

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

  if (std::strcmp(chan, "depth") == 0
      || std::strcmp(chan, "channel.depth") == 0) {
    if (t)
      *t = ANARI_FLOAT32;
    return m_channel_depth.data();
  }

  if (std::strcmp(chan, "normal") == 0
      || std::strcmp(chan, "channel.normal") == 0) {
    if (t)
      *t = ANARI_FLOAT32_VEC3;
    return m_channel_normal.data();
  }

  if (std::strcmp(chan, "albedo") == 0
      || std::strcmp(chan, "channel.albedo") == 0) {
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

  uint32_t spp = 1;
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
      pit = ro->params.find("maxRayDepth");
      if (pit != ro->params.end() && pit->second.size() == sizeof(uint32_t))
        std::memcpy(&max_depth, pit->second.data(), sizeof(uint32_t));
    }
  }

  // --- Read background, ambientColor, ambientRadiance from renderer ---
  float bg_rgba[4] = {0.f, 0.f, 0.f, 1.f};
  float ambient_color[3] = {1.f, 1.f, 1.f};
  float ambient_radiance = 0.f;
#ifdef PHOTON_HAS_OIDN
  int32_t denoise = 0;
#endif

  if (renderer_h != 0) {
    auto *ro = get((ANARIObject)renderer_h);
    if (ro) {
      auto pit = ro->params.find("background");
      if (pit != ro->params.end() && pit->second.size() == 4 * sizeof(float))
        std::memcpy(bg_rgba, pit->second.data(), 4 * sizeof(float));

      pit = ro->params.find("ambientColor");
      if (pit != ro->params.end() && pit->second.size() == 3 * sizeof(float))
        std::memcpy(ambient_color, pit->second.data(), 3 * sizeof(float));

      pit = ro->params.find("ambientRadiance");
      if (pit != ro->params.end() && pit->second.size() == sizeof(float))
        std::memcpy(&ambient_radiance, pit->second.data(), sizeof(float));

#ifdef PHOTON_HAS_OIDN
      pit = ro->params.find("denoise");
      if (pit != ro->params.end() && pit->second.size() == sizeof(int32_t))
        std::memcpy(&denoise, pit->second.data(), sizeof(int32_t));
#endif
    }
  }

  // --- Rebuild scene and accel only when world has changed ---
  const bool scene_dirty = (m_world_commit_version != m_scene_version);

  if (scene_dirty || !m_scene) {
    auto scene_opt = build_scene_from_anari((ANARIWorld)world_h, *this);
    m_scene = scene_opt ? std::move(*scene_opt) : photon::pt::SceneBuilder::make_two_quads();

    if (!m_backend)
      m_backend = photon::pt::create_best_backend();

    m_backend->build_accel(*m_scene);
    m_scene_version = m_world_commit_version;
  }

  // --- Camera is always rebuilt (cheap) ---
  photon::pt::Camera cam = build_camera_from_anari(fb, *this);

  // --- Check if accumulation needs to be reset ---
  // Compare camera fields explicitly (memcmp is unreliable due to struct padding)
  const bool camera_changed =
      cam.origin.x != m_prev_camera.origin.x
      || cam.origin.y != m_prev_camera.origin.y
      || cam.origin.z != m_prev_camera.origin.z
      || cam.lower_left.x != m_prev_camera.lower_left.x
      || cam.lower_left.y != m_prev_camera.lower_left.y
      || cam.lower_left.z != m_prev_camera.lower_left.z
      || cam.horizontal.x != m_prev_camera.horizontal.x
      || cam.horizontal.y != m_prev_camera.horizontal.y
      || cam.horizontal.z != m_prev_camera.horizontal.z
      || cam.vertical.x != m_prev_camera.vertical.x
      || cam.vertical.y != m_prev_camera.vertical.y
      || cam.vertical.z != m_prev_camera.vertical.z
      || cam.lens_radius != m_prev_camera.lens_radius
      || cam.focus_dist != m_prev_camera.focus_dist
      || cam.is_ortho != m_prev_camera.is_ortho;
  const bool size_changed = (m_fb_w != m_accum_fb_w || m_fb_h != m_accum_fb_h);
  const bool renderer_changed = (m_renderer_version != m_prev_renderer_version);
  const bool accum_dirty = scene_dirty || camera_changed || size_changed || renderer_changed;

  if (accum_dirty) {
    m_frameID = 0;
    m_prev_camera = cam;
    m_accum_fb_w = m_fb_w;
    m_accum_fb_h = m_fb_h;
    m_prev_renderer_version = m_renderer_version;
  }

  // --- Read sampleLimit from renderer params ---
  int32_t max_samples = 128;
  if (renderer_h != 0) {
    auto *ro = get((ANARIObject)renderer_h);
    if (ro) {
      auto pit = ro->params.find("sampleLimit");
      if (pit != ro->params.end() && pit->second.size() == sizeof(int32_t))
        std::memcpy(&max_samples, pit->second.data(), sizeof(int32_t));
    }
  }

  // --- Cap accumulation (stop rendering once we reach the limit) ---
  // sampleLimit == 0 means unlimited
  if (max_samples > 0 && m_frameID >= max_samples) {
    // Already converged — just reuse the existing averaged buffers
    m_lastDuration = 0.f;
    return;
  }

  // --- Render with cached backend + scene ---
  auto t0 = std::chrono::steady_clock::now();

  photon::pt::PathTracer pt;
  pt.params.width = m_fb_w;
  pt.params.height = m_fb_h;
  pt.params.samples_per_pixel = spp;
  pt.params.max_depth = max_depth;
  pt.params.sample_offset = uint32_t(m_frameID);
  pt.params.background_color = {bg_rgba[0], bg_rgba[1], bg_rgba[2]};
  pt.params.background_alpha = bg_rgba[3];
  pt.params.ambient_color = {ambient_color[0], ambient_color[1], ambient_color[2]};
  pt.params.ambient_radiance = ambient_radiance;
  pt.camera = cam;
  pt.set_scene(*m_scene);
  pt.set_backend_ref(*m_backend);

  auto result = pt.render();

  auto t1 = std::chrono::steady_clock::now();
  m_lastDuration = std::chrono::duration<float>(t1 - t0).count();

  auto color_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.color);
  auto depth_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.depth);
  auto normal_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.normal);
  auto albedo_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.albedo);

  const size_t num_pixels = size_t(m_fb_w) * size_t(m_fb_h);

  // Resize accumulation and output buffers
  m_accumColor.resize(num_pixels * 4, 0.f);
  m_accumDepth.resize(num_pixels, 0.f);
  m_accumNormal.resize(num_pixels * 3, 0.f);
  m_accumAlbedo.resize(num_pixels * 3, 0.f);

  m_channel_color.resize(num_pixels * 4);
  m_channel_depth.resize(num_pixels);
  m_channel_normal.resize(num_pixels * 3);
  m_channel_albedo.resize(num_pixels * 3);

  // If accumulation was reset, clear the buffers
  if (m_frameID == 0) {
    std::fill(m_accumColor.begin(), m_accumColor.end(), 0.f);
    std::fill(m_accumDepth.begin(), m_accumDepth.end(), 0.f);
    std::fill(m_accumNormal.begin(), m_accumNormal.end(), 0.f);
    std::fill(m_accumAlbedo.begin(), m_accumAlbedo.end(), 0.f);
  }

  // Add this frame's contribution to the accumulation buffers
  for (uint32_t y = 0; y < m_fb_h; ++y) {
    for (uint32_t x = 0; x < m_fb_w; ++x) {
      const size_t idx = size_t(y) * size_t(m_fb_w) + x;
      const uint32_t src_y = m_fb_h - 1 - y;

      const auto c = color_host(src_y, x);
      m_accumColor[4 * idx + 0] += c.x;
      m_accumColor[4 * idx + 1] += c.y;
      m_accumColor[4 * idx + 2] += c.z;
      m_accumColor[4 * idx + 3] += 1.f;

      m_accumDepth[idx] += depth_host(src_y, x);

      const auto n = normal_host(src_y, x);
      m_accumNormal[3 * idx + 0] += n.x;
      m_accumNormal[3 * idx + 1] += n.y;
      m_accumNormal[3 * idx + 2] += n.z;

      const auto a = albedo_host(src_y, x);
      m_accumAlbedo[3 * idx + 0] += a.x;
      m_accumAlbedo[3 * idx + 1] += a.y;
      m_accumAlbedo[3 * idx + 2] += a.z;
    }
  }

  m_frameID += spp;

  // Write averaged result to output buffers
  const float inv = 1.f / float(m_frameID);
  for (size_t idx = 0; idx < num_pixels; ++idx) {
    const auto cr = photon::pt::clamp01({
        m_accumColor[4 * idx + 0] * inv,
        m_accumColor[4 * idx + 1] * inv,
        m_accumColor[4 * idx + 2] * inv});
    out[4 * idx + 0] = cr.x;
    out[4 * idx + 1] = cr.y;
    out[4 * idx + 2] = cr.z;
    out[4 * idx + 3] = 1.f;

    m_channel_color[4 * idx + 0] = cr.x;
    m_channel_color[4 * idx + 1] = cr.y;
    m_channel_color[4 * idx + 2] = cr.z;
    m_channel_color[4 * idx + 3] = 1.f;

    m_channel_depth[idx] = m_accumDepth[idx] * inv;

    m_channel_normal[3 * idx + 0] = m_accumNormal[3 * idx + 0] * inv;
    m_channel_normal[3 * idx + 1] = m_accumNormal[3 * idx + 1] * inv;
    m_channel_normal[3 * idx + 2] = m_accumNormal[3 * idx + 2] * inv;

    m_channel_albedo[3 * idx + 0] = m_accumAlbedo[3 * idx + 0] * inv;
    m_channel_albedo[3 * idx + 1] = m_accumAlbedo[3 * idx + 1] * inv;
    m_channel_albedo[3 * idx + 2] = m_accumAlbedo[3 * idx + 2] * inv;
  }

#ifdef PHOTON_HAS_OIDN
  // --- Denoise the averaged result if requested ---
  if (denoise) {
    // Extract RGB from RGBA channel buffer (OIDN needs packed RGB)
    m_denoise_rgb.resize(num_pixels * 3);
    for (size_t idx = 0; idx < num_pixels; ++idx) {
      m_denoise_rgb[3 * idx + 0] = m_channel_color[4 * idx + 0];
      m_denoise_rgb[3 * idx + 1] = m_channel_color[4 * idx + 1];
      m_denoise_rgb[3 * idx + 2] = m_channel_color[4 * idx + 2];
    }

    // Denoise in-place — albedo and normal are already packed RGB
    m_denoiser.denoise_buffer(m_denoise_rgb.data(),
                              m_channel_albedo.data(),
                              m_channel_normal.data(),
                              m_fb_w, m_fb_h);

    // Write denoised RGB back to RGBA output
    for (size_t idx = 0; idx < num_pixels; ++idx) {
      out[4 * idx + 0] = m_denoise_rgb[3 * idx + 0];
      out[4 * idx + 1] = m_denoise_rgb[3 * idx + 1];
      out[4 * idx + 2] = m_denoise_rgb[3 * idx + 2];
      out[4 * idx + 3] = 1.f;

      m_channel_color[4 * idx + 0] = m_denoise_rgb[3 * idx + 0];
      m_channel_color[4 * idx + 1] = m_denoise_rgb[3 * idx + 1];
      m_channel_color[4 * idx + 2] = m_denoise_rgb[3 * idx + 2];
      m_channel_color[4 * idx + 3] = 1.f;
    }
  }
#endif
}

int PhotonDevice::frameReady(ANARIFrame, ANARIWaitMask) { return 1; }

void PhotonDevice::report(ANARIObject, ANARIDataType, ANARIStatusSeverity, ANARIStatusCode, const char *) {}

void PhotonDevice::discardFrame(ANARIFrame) {}

}
