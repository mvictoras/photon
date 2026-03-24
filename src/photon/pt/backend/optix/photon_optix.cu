#include <optix.h>
#include <optix_device.h>

struct MeshData {
  float3 *positions;
  unsigned int *indices;
  float3 *normals;
  float2 *texcoords;
  unsigned int *material_ids;
  float3 *vertex_colors;
  unsigned int has_normals;
  unsigned int has_texcoords;
  unsigned int has_material_ids;
  unsigned int has_vertex_colors;
};

struct HitResultGPU {
  float t;
  float3 position;
  float3 normal;
  float3 shading_normal;
  float2 uv;
  float3 interpolated_color;
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
  void *hit_results;
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
  unsigned int use_sbt_data;
  MeshData *object_meshes;
  unsigned int object_count;
};

extern "C" __constant__ Params params;

__device__ inline float3 normalize3(float3 v){float l=sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);if(l>0.f){float i=1.f/l;v.x*=i;v.y*=i;v.z*=i;}return v;}
__device__ inline float3 cross3(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ inline float dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}

__device__ void shade_hit(unsigned int pid,float bu,float bv,
    float3 ro,float3 rd,float t,
    float3*pos,unsigned int*idx,float3*nrm,float2*uv_arr,
    unsigned int*mat_ids,float3*vcols,
    unsigned int hn,unsigned int huv,unsigned int hm,unsigned int hvc,
    HitResultGPU&hr)
{
  const unsigned int i0=idx[pid*3],i1=idx[pid*3+1],i2=idx[pid*3+2];
  const float bw=1.f-bu-bv;
  const float3 p0=pos[i0],p1=pos[i1],p2=pos[i2];
  float3 e1={p1.x-p0.x,p1.y-p0.y,p1.z-p0.z},e2={p2.x-p0.x,p2.y-p0.y,p2.z-p0.z};
  float3 ng=normalize3(cross3(e1,e2));
  hr.position={ro.x+rd.x*t,ro.y+rd.y*t,ro.z+rd.z*t};
  hr.normal={ng.x,ng.y,ng.z};hr.prim_id=pid;hr.geom_id=0;hr.t=t;
  if(hn){float3 n0=nrm[i0],n1=nrm[i1],n2=nrm[i2];
    float3 sn=normalize3(make_float3(n0.x*bw+n1.x*bu+n2.x*bv,n0.y*bw+n1.y*bu+n2.y*bv,n0.z*bw+n1.z*bu+n2.z*bv));
    if(dot3(sn,ng)<0.f){sn.x=-sn.x;sn.y=-sn.y;sn.z=-sn.z;}
    hr.shading_normal={sn.x,sn.y,sn.z};
  } else hr.shading_normal=hr.normal;
  if(huv){float2 t0=uv_arr[i0],t1=uv_arr[i1],t2=uv_arr[i2];
    hr.uv={t0.x*bw+t1.x*bu+t2.x*bv,t0.y*bw+t1.y*bu+t2.y*bv};
  } else hr.uv={bu,bv};
  hr.material_id=hm?mat_ids[pid]:0u;
  hr.has_interpolated_color=false;
  if(hvc){float3 c0=vcols[i0];
    if(c0.x>=0.f){float3 c1=vcols[i1],c2=vcols[i2];
      hr.interpolated_color={c0.x*bw+c1.x*bu+c2.x*bv,c0.y*bw+c1.y*bu+c2.y*bv,c0.z*bw+c1.z*bu+c2.z*bv};
      hr.has_interpolated_color=true;
    }
  }
}

extern "C" __global__ void __raygen__rg()
{
  const unsigned int idx=optixGetLaunchIndex().x;
  if(idx>=params.count) return;
  const float3 org=params.origins[idx];
  const float3 dir=params.directions[idx];
  unsigned int p0=0,p1=0,p2=0,p3=0,p4=0,p5=0;
  optixTrace(params.handle,org,dir,params.tmin[idx],params.tmax[idx],
      0.f,0xFFu,OPTIX_RAY_FLAG_NONE,0,1,0,p0,p1,p2,p3,p4,p5);
  if(params.mode==1){params.occluded[idx]=p0;return;}
  HitResultGPU hr{};
  hr.hit=(p0!=0); hr.inst_id=p5;
  if(hr.hit){
    const float bu=__uint_as_float(p3),bv=__uint_as_float(p4);
    const float t=__uint_as_float(p2);
    if(params.use_sbt_data && params.object_meshes && p5 < params.object_count){
      const MeshData &md=params.object_meshes[p5];
      shade_hit(p1,bu,bv,org,dir,t,
          md.positions,md.indices,md.normals,md.texcoords,
          md.material_ids,md.vertex_colors,
          md.has_normals,md.has_texcoords,md.has_material_ids,md.has_vertex_colors,hr);
    } else {
      shade_hit(p1,bu,bv,org,dir,t,
          params.mesh_positions,params.mesh_indices,params.mesh_normals,params.mesh_texcoords,
          params.mesh_material_ids,params.mesh_vertex_colors,
          params.has_normals,params.has_texcoords,params.has_material_ids,params.has_vertex_colors,hr);
    }
  } else {
    hr.t=0.f;hr.prim_id=0;hr.material_id=0;
    hr.uv={0.f,0.f};hr.position=hr.normal=hr.shading_normal=hr.interpolated_color={0.f,0.f,0.f};
    hr.has_interpolated_color=false;
  }
  ((HitResultGPU*)params.hit_results)[idx]=hr;
}

extern "C" __global__ void __miss__ms() { optixSetPayload_0(0); }

extern "C" __global__ void __closesthit__ch()
{
  optixSetPayload_0(1);
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
  const float2 b=optixGetTriangleBarycentrics();
  optixSetPayload_3(__float_as_uint(b.x));
  optixSetPayload_4(__float_as_uint(b.y));
  optixSetPayload_5(optixGetInstanceId());
}
