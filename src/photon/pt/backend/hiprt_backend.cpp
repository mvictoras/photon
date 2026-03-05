#ifdef PHOTON_HAS_HIPRT

#include "photon/pt/backend/hiprt_backend.h"

#include "photon/pt/scene.h"

#include <stdexcept>

namespace photon::pt {

HiprtBackend::HiprtBackend()
{
  hiprtContextCreationInput input{};
  input.deviceType = hiprtDeviceAMD;
  hiprtResult r = hiprtCreateContext(hiprtApiVersion, input, m_context);
  if (r != hiprtSuccess) {
    throw std::runtime_error("hiprtCreateContext failed");
  }
}

HiprtBackend::~HiprtBackend()
{
  if (m_scene_handle) {
    hiprtDestroyScene(m_context, m_scene_handle);
    m_scene_handle = nullptr;
  }
  if (m_geometry) {
    hiprtDestroyGeometry(m_context, m_geometry);
    m_geometry = nullptr;
  }
  if (m_context) {
    hiprtDestroyContext(m_context);
    m_context = nullptr;
  }
}

void HiprtBackend::build_accel(const Scene &scene)
{
  m_mesh = scene.mesh;

  throw std::runtime_error("HiprtBackend::build_accel not implemented yet");
}

void HiprtBackend::trace_closest(const RayBatch &, HitBatch &)
{
  throw std::runtime_error("HiprtBackend::trace_closest not implemented yet");
}

void HiprtBackend::trace_occluded(const RayBatch &, Kokkos::View<u32 *>)
{
  throw std::runtime_error("HiprtBackend::trace_occluded not implemented yet");
}

}

#endif
