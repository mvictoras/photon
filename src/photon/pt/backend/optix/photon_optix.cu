#include <optix.h>
#include <optix_device.h>

struct Vec3f { float x, y, z; };
struct Vec2f { float x, y; };

struct HitResultGPU {
  float t;
  Vec3f position;
  Vec3f normal;
  Vec3f shading_normal;
  Vec2f uv;
  Vec3f interpolated_color;
  unsigned int prim_id;
  unsigned int geom_id;
  unsigned int inst_id;
  unsigned int material_id;
  bool hit;
  bool has_interpolated_color;
};

struct Params {
  OptixTraversableHandle handle;
  float3 *origins;
  float3 *directions;
  float *tmin;
  float *tmax;

  HitResultGPU *hit_results;
  unsigned int *occluded;

  float3 *mesh_positions;
  unsigned int *mesh_indices;
  float3 *mesh_normals;
  float2 *mesh_texcoords;
  unsigned int *mesh_material_ids;
  float3 *mesh_vertex_colors;

  unsigned int has_normals;
  unsigned int has_texcoords;
  unsigned int has_material_ids;
  unsigned int has_vertex_colors;

  unsigned int count;
  unsigned int mode;
};

extern "C" __constant__ Params params;

__device__ inline float3 normalize3(float3 v)
{
  float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
  if (len > 0.f) { float inv = 1.f / len; v.x *= inv; v.y *= inv; v.z *= inv; }
  return v;
}

__device__ inline float3 cross3(float3 a, float3 b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ inline float dot3(float3 a, float3 b)
{
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

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

  if (params.mode == 1) {
    params.occluded[idx] = p0;
    return;
  }

  HitResultGPU hr;
  hr.hit = (p0 != 0);
  hr.geom_id = 0;
  hr.inst_id = 0;
  hr.has_interpolated_color = false;
  hr.interpolated_color = {0.f, 0.f, 0.f};

  if (hr.hit) {
    hr.prim_id = p1;
    hr.t = __uint_as_float(p2);
    const float bu = __uint_as_float(p3);
    const float bv = __uint_as_float(p4);
    const float bw = 1.f - bu - bv;

    hr.position = {org.x + dir.x * hr.t, org.y + dir.y * hr.t, org.z + dir.z * hr.t};

    const unsigned int pid = hr.prim_id;
    const unsigned int i0 = params.mesh_indices[pid * 3 + 0];
    const unsigned int i1 = params.mesh_indices[pid * 3 + 1];
    const unsigned int i2 = params.mesh_indices[pid * 3 + 2];

    const float3 p0v = params.mesh_positions[i0];
    const float3 p1v = params.mesh_positions[i1];
    const float3 p2v = params.mesh_positions[i2];

    float3 e1 = {p1v.x-p0v.x, p1v.y-p0v.y, p1v.z-p0v.z};
    float3 e2 = {p2v.x-p0v.x, p2v.y-p0v.y, p2v.z-p0v.z};
    float3 ng = normalize3(cross3(e1, e2));
    hr.normal = {ng.x, ng.y, ng.z};

    if (params.has_normals) {
      float3 n0 = params.mesh_normals[i0];
      float3 n1 = params.mesh_normals[i1];
      float3 n2 = params.mesh_normals[i2];
      float3 sn = normalize3(make_float3(
          n0.x*bw + n1.x*bu + n2.x*bv,
          n0.y*bw + n1.y*bu + n2.y*bv,
          n0.z*bw + n1.z*bu + n2.z*bv));
      if (dot3(sn, ng) < 0.f) { sn.x = -sn.x; sn.y = -sn.y; sn.z = -sn.z; }
      hr.shading_normal = {sn.x, sn.y, sn.z};
    } else {
      hr.shading_normal = hr.normal;
    }

    if (params.has_texcoords) {
      float2 t0 = params.mesh_texcoords[i0];
      float2 t1 = params.mesh_texcoords[i1];
      float2 t2 = params.mesh_texcoords[i2];
      hr.uv = {t0.x*bw + t1.x*bu + t2.x*bv, t0.y*bw + t1.y*bu + t2.y*bv};
    } else {
      hr.uv = {bu, bv};
    }

    hr.material_id = params.has_material_ids ? params.mesh_material_ids[pid] : 0u;

    if (params.has_vertex_colors) {
      float3 c0 = params.mesh_vertex_colors[i0];
      if (c0.x >= 0.f) {
        float3 c1 = params.mesh_vertex_colors[i1];
        float3 c2 = params.mesh_vertex_colors[i2];
        hr.interpolated_color = {c0.x*bw+c1.x*bu+c2.x*bv, c0.y*bw+c1.y*bu+c2.y*bv, c0.z*bw+c1.z*bu+c2.z*bv};
        hr.has_interpolated_color = true;
      }
    }
  } else {
    hr.t = 0.f;
    hr.prim_id = 0;
    hr.material_id = 0;
    hr.uv = {0.f, 0.f};
    hr.position = {0.f, 0.f, 0.f};
    hr.normal = {0.f, 0.f, 0.f};
    hr.shading_normal = {0.f, 0.f, 0.f};
  }

  params.hit_results[idx] = hr;
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
