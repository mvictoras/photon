#pragma once

#include <anari/backend/DeviceImpl.h>
#include <anari/anari.h>

#include <helium/utility/IntrusivePtr.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "photon/pt/denoiser.h"

#include "photon/pt/camera.h"
#include "photon/pt/scene.h"
#include "photon/pt/backend/ray_backend.h"

namespace photon::anari_device {

struct PhotonDevice final : public anari::DeviceImpl, public helium::RefCounted
{
  explicit PhotonDevice(ANARILibrary library);

  ANARIArray1D newArray1D(const void *appMemory, ANARIMemoryDeleter deleter, const void *userData, ANARIDataType,
      uint64_t numItems1) override;
  ANARIArray2D newArray2D(const void *appMemory, ANARIMemoryDeleter deleter, const void *userData, ANARIDataType,
      uint64_t numItems1, uint64_t numItems2) override;
  ANARIArray3D newArray3D(const void *appMemory, ANARIMemoryDeleter deleter, const void *userData, ANARIDataType,
      uint64_t numItems1, uint64_t numItems2, uint64_t numItems3) override;
  void *mapArray(ANARIArray) override;
  void unmapArray(ANARIArray) override;

  ANARIGeometry newGeometry(const char *type) override;
  ANARISurface newSurface() override;
  ANARIWorld newWorld() override;
  ANARIRenderer newRenderer(const char *type) override;
  ANARIFrame newFrame() override;

  ANARILight newLight(const char *) override;
  ANARICamera newCamera(const char *) override;
  ANARISpatialField newSpatialField(const char *) override;
  ANARIVolume newVolume(const char *) override;
  ANARIMaterial newMaterial(const char *) override;
  ANARISampler newSampler(const char *) override;
  ANARIGroup newGroup() override;
  ANARIInstance newInstance(const char *) override;

  const char **getObjectSubtypes(ANARIDataType objectType) override;
  const void *getObjectInfo(ANARIDataType objectType, const char *objectSubtype, const char *infoName,
      ANARIDataType infoType) override;
  const void *getParameterInfo(ANARIDataType objectType, const char *objectSubtype, const char *parameterName,
      ANARIDataType parameterType, const char *infoName, ANARIDataType infoType) override;

  int getProperty(ANARIObject object, const char *name, ANARIDataType type, void *mem, uint64_t size,
      ANARIWaitMask mask) override;
  void setParameter(ANARIObject object, const char *name, ANARIDataType type, const void *mem) override;
  void unsetParameter(ANARIObject object, const char *name) override;
  void unsetAllParameters(ANARIObject object) override;
  void commitParameters(ANARIObject object) override;
  void release(ANARIObject object) override;
  void retain(ANARIObject object) override;

  void *mapParameterArray1D(ANARIObject, const char *, ANARIDataType, uint64_t, uint64_t *) override;
  void *mapParameterArray2D(ANARIObject, const char *, ANARIDataType, uint64_t, uint64_t, uint64_t *) override;
  void *mapParameterArray3D(ANARIObject, const char *, ANARIDataType, uint64_t, uint64_t, uint64_t, uint64_t *) override;
  void unmapParameterArray(ANARIObject, const char *) override;

  const void *frameBufferMap(ANARIFrame fb, const char *channel, uint32_t *width, uint32_t *height,
      ANARIDataType *pixelType) override;
  void frameBufferUnmap(ANARIFrame fb, const char *channel) override;
  void renderFrame(ANARIFrame) override;
  int frameReady(ANARIFrame, ANARIWaitMask) override;
  void discardFrame(ANARIFrame) override;

  void report(ANARIObject source, ANARIDataType sourceType, ANARIStatusSeverity severity, ANARIStatusCode code,
      const char *msg);

  public:
  struct Object

  {
    ANARIDataType object_type{ANARI_UNKNOWN};
    int64_t refcount{1};

    const void *memory{nullptr};
    const void *userdata{nullptr};
    ANARIMemoryDeleter deleter{nullptr};

    ANARIDataType array_element_type{ANARI_UNKNOWN};
    uint64_t array_num_items1{0};
    uint64_t array_num_items2{0};
    uint64_t array_num_items3{0};

    std::unordered_map<std::string, std::vector<std::byte>> params;
  };

  Object *getObject(uintptr_t handle) const { return const_cast<PhotonDevice *>(this)->get((ANARIObject)handle); }

 private:

  uintptr_t alloc_handle(ANARIDataType t);
  Object *get(ANARIObject o);



   uint64_t m_next_id{1};
   std::map<uintptr_t, std::unique_ptr<Object>> m_objects;
    std::vector<float> m_framebuffer;
    std::vector<char> m_fb_bytes;
    uint32_t m_fb_w{0}, m_fb_h{0};

    std::vector<float> m_channel_color;
    std::vector<float> m_channel_depth;
    std::vector<float> m_channel_normal;
    std::vector<float> m_channel_albedo;

    // Cached rendering state — persist across frames
    std::unique_ptr<photon::pt::RayBackend> m_backend;
    std::optional<photon::pt::Scene> m_scene;
    uint64_t m_scene_version{0};       // version when scene was last built
    uint64_t m_world_commit_version{0}; // bumped on scene-affecting commits
    uint64_t m_renderer_version{0};     // bumped on renderer commits
    uint64_t m_prev_renderer_version{0}; // renderer version at last frame

    // Accumulation state
    int m_frameID{0};                    // accumulated sample count
    std::vector<float> m_accumColor;     // running sum (RGBA per pixel)
    std::vector<float> m_accumDepth;     // running sum (1 float per pixel)
    std::vector<float> m_accumNormal;    // running sum (3 floats per pixel)
    std::vector<float> m_accumAlbedo;    // running sum (3 floats per pixel)
    float m_lastDuration{0.f};           // seconds for last renderFrame
    uint32_t m_accum_fb_w{0}, m_accum_fb_h{0}; // resolution when accum started
    photon::pt::Camera m_prev_camera{};  // camera from previous frame

    // Denoiser (persistent across frames)
#ifdef PHOTON_HAS_OIDN
    photon::pt::Denoiser m_denoiser;
    std::vector<float> m_denoise_rgb;    // temp RGB buffer for denoiser
#endif
};

}
