#include <optix.h>
#include <optix_device.h>

struct Params {
  OptixTraversableHandle handle;
  float3 *origins;
  float3 *directions;
  float *tmin;
  float *tmax;
  float *hit_t;
  unsigned int *hit_prim_id;
  unsigned int *hit_flag;
  float *hit_bary_u;
  float *hit_bary_v;
  unsigned int *occluded;
  unsigned int count;
  unsigned int mode;
};

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
  const unsigned int idx = optixGetLaunchIndex().x;
  if (idx >= params.count)
    return;

  const float3 org = params.origins[idx];
  const float3 dir = params.directions[idx];
  const float tn = params.tmin[idx];
  const float tf = params.tmax[idx];

  unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;

  optixTrace(
      params.handle,
      org, dir,
      tn, tf,
      0.0f,
      0xFFu,
      OPTIX_RAY_FLAG_NONE,
      0, 1, 0,
      p0, p1, p2, p3, p4);

  if (params.mode == 0) {
    params.hit_flag[idx] = p0;
    if (p0) {
      params.hit_prim_id[idx] = p1;
      params.hit_t[idx] = __uint_as_float(p2);
      params.hit_bary_u[idx] = __uint_as_float(p3);
      params.hit_bary_v[idx] = __uint_as_float(p4);
    }
  } else {
    params.occluded[idx] = p0;
  }
}

extern "C" __global__ void __miss__ms()
{
  optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__ch()
{
  optixSetPayload_0(1);
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2(__float_as_uint(optixGetRayTmax()));

  const float2 bary = optixGetTriangleBarycentrics();
  optixSetPayload_3(__float_as_uint(bary.x));
  optixSetPayload_4(__float_as_uint(bary.y));
}
