#ifdef PHOTON_HAS_OPTIX

#include "photon/pt/backend/optix_backend.h"

#include "photon/pt/scene.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace photon::pt {
namespace {

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

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct EmptyData {};

static void optix_log_callback(unsigned int level, const char *tag, const char *message, void *)
{
  if (level >= 4)
    std::fprintf(stderr, "[photon][optix][%s] %s\n", tag ? tag : "", message ? message : "");
}

inline void check_optix(const OptixResult r, const char *what)
{
  if (r != OPTIX_SUCCESS)
    throw std::runtime_error(std::string("OptiX error ") + what + ": " + std::to_string(int(r)));
}

inline void check_cuda(const cudaError_t e, const char *what)
{
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error ") + what + ": " + cudaGetErrorString(e));
}

}

OptixBackend::OptixBackend()
{
  check_cuda(cudaFree(nullptr), "init cuda");

  CUcontext cu_ctx = 0;
  OptixDeviceContextOptions options{};
  options.logCallbackFunction = optix_log_callback;
  options.logCallbackLevel = 4;
  check_optix(optixInit(), "optixInit");
  check_optix(optixDeviceContextCreate(cu_ctx, &options, &m_context), "optixDeviceContextCreate");
}

OptixBackend::~OptixBackend()
{
  if (m_pipeline) {
    optixPipelineDestroy(m_pipeline);
    m_pipeline = nullptr;
  }
  if (m_module) {
    optixModuleDestroy(m_module);
    m_module = nullptr;
  }
  if (m_gas_buffer) {
    cudaFree(reinterpret_cast<void *>(m_gas_buffer));
    m_gas_buffer = 0;
  }
  if (m_context) {
    optixDeviceContextDestroy(m_context);
    m_context = nullptr;
  }
}

void OptixBackend::create_pipeline()
{
  if (m_pipeline)
    return;

  OptixPipelineCompileOptions pipeline_compile_options{};
  pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.usesMotionBlur = 0;
  pipeline_compile_options.numPayloadValues = 5;
  pipeline_compile_options.numAttributeValues = 2;
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  OptixModuleCompileOptions module_compile_options{};
  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  extern const char photon_optix_ptx[];
  extern const unsigned long long photon_optix_ptx_size;

  char log[2048]{};
  size_t log_size = sizeof(log);

  check_optix(optixModuleCreate(
                  m_context,
                  &module_compile_options,
                  &pipeline_compile_options,
                  photon_optix_ptx,
                  photon_optix_ptx_size,
                  log,
                  &log_size,
                  &m_module),
              "optixModuleCreate");

  OptixProgramGroupOptions pg_options{};

  OptixProgramGroupDesc raygen_desc{};
  raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_desc.raygen.module = m_module;
  raygen_desc.raygen.entryFunctionName = "__raygen__rg";

  OptixProgramGroupDesc miss_desc{};
  miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_desc.miss.module = m_module;
  miss_desc.miss.entryFunctionName = "__miss__ms";

  OptixProgramGroupDesc hit_desc{};
  hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_desc.hitgroup.moduleCH = m_module;
  hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  hit_desc.hitgroup.moduleAH = nullptr;
  hit_desc.hitgroup.entryFunctionNameAH = nullptr;

  OptixProgramGroup raygen_pg = nullptr;
  OptixProgramGroup miss_pg = nullptr;
  OptixProgramGroup hit_pg = nullptr;

  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &raygen_desc, 1, &pg_options, log, &log_size, &raygen_pg),
              "ProgramGroupCreate raygen");
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &miss_desc, 1, &pg_options, log, &log_size, &miss_pg),
              "ProgramGroupCreate miss");
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &hit_desc, 1, &pg_options, log, &log_size, &hit_pg),
              "ProgramGroupCreate hit");

  OptixProgramGroup groups[] = {raygen_pg, miss_pg, hit_pg};

  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;

  log_size = sizeof(log);
  check_optix(optixPipelineCreate(
                  m_context,
                  &pipeline_compile_options,
                  &link_options,
                  groups,
                  3,
                  log,
                  &log_size,
                  &m_pipeline),
              "optixPipelineCreate");

  check_optix(optixPipelineSetStackSize(m_pipeline, 2048, 2048, 2048, 1),
              "optixPipelineSetStackSize");

  SbtRecord<EmptyData> rg_record{};
  SbtRecord<EmptyData> ms_record{};
  SbtRecord<EmptyData> hg_record{};

  optixSbtRecordPackHeader(raygen_pg, &rg_record);
  optixSbtRecordPackHeader(miss_pg, &ms_record);
  optixSbtRecordPackHeader(hit_pg, &hg_record);

  CUdeviceptr d_rg = 0, d_ms = 0, d_hg = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rg_record)), "malloc sbt rg");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ms), sizeof(ms_record)), "malloc sbt ms");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_hg), sizeof(hg_record)), "malloc sbt hg");

  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rg_record, sizeof(rg_record), cudaMemcpyHostToDevice), "sbt rg");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_ms), &ms_record, sizeof(ms_record), cudaMemcpyHostToDevice), "sbt ms");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_hg), &hg_record, sizeof(hg_record), cudaMemcpyHostToDevice), "sbt hg");

  m_sbt.raygenRecord = d_rg;
  m_sbt.missRecordBase = d_ms;
  m_sbt.missRecordStrideInBytes = sizeof(ms_record);
  m_sbt.missRecordCount = 1;
  m_sbt.hitgroupRecordBase = d_hg;
  m_sbt.hitgroupRecordStrideInBytes = sizeof(hg_record);
  m_sbt.hitgroupRecordCount = 1;

  optixProgramGroupDestroy(raygen_pg);
  optixProgramGroupDestroy(miss_pg);
  optixProgramGroupDestroy(hit_pg);
}

void OptixBackend::build_accel(const Scene &scene)
{
  m_mesh = scene.mesh;

  auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.positions);
  auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.indices);

  const size_t vertex_count = pos_h.extent(0);
  const size_t tri_count = idx_h.extent(0) / 3;

  std::vector<float3> vertices(vertex_count);
  for (size_t i = 0; i < vertex_count; ++i)
    vertices[i] = make_float3(pos_h(i).x, pos_h(i).y, pos_h(i).z);

  std::vector<uint3> indices(tri_count);
  for (size_t i = 0; i < tri_count; ++i)
    indices[i] = make_uint3(idx_h(i * 3 + 0), idx_h(i * 3 + 1), idx_h(i * 3 + 2));

  CUdeviceptr d_vertices = 0, d_indices = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices.size() * sizeof(float3)), "malloc verts");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_indices), indices.size() * sizeof(uint3)), "malloc idx");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_vertices), vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice), "copy verts");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_indices), indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice), "copy idx");

  OptixBuildInput build_input{};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  auto &tri = build_input.triangleArray;
  tri.vertexBuffers = &d_vertices;
  tri.numVertices = static_cast<unsigned>(vertex_count);
  tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  tri.vertexStrideInBytes = sizeof(float3);
  tri.indexBuffer = d_indices;
  tri.numIndexTriplets = static_cast<unsigned>(tri_count);
  tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  tri.indexStrideInBytes = sizeof(uint3);
  unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
  tri.flags = &flags;
  tri.numSbtRecords = 1;

  OptixAccelBuildOptions accel_options{};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes buffer_sizes{};
  check_optix(optixAccelComputeMemoryUsage(m_context, &accel_options, &build_input, 1, &buffer_sizes), "accel memory");

  CUdeviceptr d_temp = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_temp), buffer_sizes.tempSizeInBytes), "malloc temp");

  if (m_gas_buffer) {
    cudaFree(reinterpret_cast<void *>(m_gas_buffer));
    m_gas_buffer = 0;
  }
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&m_gas_buffer), buffer_sizes.outputSizeInBytes), "malloc gas");

  CUdeviceptr d_compacted_size = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_compacted_size), sizeof(size_t)), "malloc compact sz");
  OptixAccelEmitDesc emit_desc{};
  emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_desc.result = d_compacted_size;

  check_optix(optixAccelBuild(m_context, 0, &accel_options, &build_input, 1,
                              d_temp, buffer_sizes.tempSizeInBytes,
                              m_gas_buffer, buffer_sizes.outputSizeInBytes,
                              &m_gas_handle, &emit_desc, 1),
              "optixAccelBuild");

  cudaFree(reinterpret_cast<void *>(d_temp));

  size_t compacted_size = 0;
  cudaMemcpy(&compacted_size, reinterpret_cast<void *>(d_compacted_size), sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaFree(reinterpret_cast<void *>(d_compacted_size));

  if (compacted_size > 0 && compacted_size < buffer_sizes.outputSizeInBytes) {
    CUdeviceptr d_compacted = 0;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_compacted), compacted_size), "malloc compact");
    check_optix(optixAccelCompact(m_context, 0, m_gas_handle, d_compacted, compacted_size, &m_gas_handle), "compact");
    cudaFree(reinterpret_cast<void *>(m_gas_buffer));
    m_gas_buffer = d_compacted;
  }

  cudaFree(reinterpret_cast<void *>(d_vertices));
  cudaFree(reinterpret_cast<void *>(d_indices));

  std::fprintf(stderr, "[photon] OptiX GAS built: %zu tris, %.1f KB\n",
               tri_count, float(compacted_size > 0 ? compacted_size : buffer_sizes.outputSizeInBytes) / 1024.f);

  create_pipeline();
}

void OptixBackend::trace_closest(const RayBatch &rays, HitBatch &hits)
{
  const u32 n = rays.count;

  float *d_hit_t = nullptr;
  unsigned int *d_hit_prim_id = nullptr;
  unsigned int *d_hit_flag = nullptr;
  float *d_hit_bary_u = nullptr;
  float *d_hit_bary_v = nullptr;

  check_cuda(cudaMalloc(&d_hit_t, n * sizeof(float)), "malloc hit_t");
  check_cuda(cudaMalloc(&d_hit_prim_id, n * sizeof(unsigned int)), "malloc hit_prim_id");
  check_cuda(cudaMalloc(&d_hit_flag, n * sizeof(unsigned int)), "malloc hit_flag");
  check_cuda(cudaMalloc(&d_hit_bary_u, n * sizeof(float)), "malloc hit_bary_u");
  check_cuda(cudaMalloc(&d_hit_bary_v, n * sizeof(float)), "malloc hit_bary_v");
  check_cuda(cudaMemset(d_hit_flag, 0, n * sizeof(unsigned int)), "memset hit_flag");

  // Vec3 and float3 share identical layout — safe to reinterpret
  float3 *d_origins = reinterpret_cast<float3 *>(rays.origins.data());
  float3 *d_directions = reinterpret_cast<float3 *>(rays.directions.data());
  float *d_tmin = rays.tmin.data();
  float *d_tmax = rays.tmax.data();

  Params params{};
  params.handle = m_gas_handle;
  params.origins = d_origins;
  params.directions = d_directions;
  params.tmin = d_tmin;
  params.tmax = d_tmax;
  params.hit_t = d_hit_t;
  params.hit_prim_id = d_hit_prim_id;
  params.hit_flag = d_hit_flag;
  params.hit_bary_u = d_hit_bary_u;
  params.hit_bary_v = d_hit_bary_v;
  params.occluded = nullptr;
  params.count = n;
  params.mode = 0;

  CUdeviceptr d_params = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)), "malloc params");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice), "copy params");

  check_optix(optixLaunch(m_pipeline, 0, d_params, sizeof(Params), &m_sbt, n, 1, 1), "optixLaunch closest");
  check_cuda(cudaDeviceSynchronize(), "sync after launch");

  std::vector<float> h_hit_t(n);
  std::vector<float> h_bary_u(n);
  std::vector<float> h_bary_v(n);
  std::vector<unsigned int> h_hit_prim_id(n);
  std::vector<unsigned int> h_hit_flag(n);

  check_cuda(cudaMemcpy(h_hit_t.data(), d_hit_t, n * sizeof(float), cudaMemcpyDeviceToHost), "copy hit_t");
  check_cuda(cudaMemcpy(h_bary_u.data(), d_hit_bary_u, n * sizeof(float), cudaMemcpyDeviceToHost), "copy bary_u");
  check_cuda(cudaMemcpy(h_bary_v.data(), d_hit_bary_v, n * sizeof(float), cudaMemcpyDeviceToHost), "copy bary_v");
  check_cuda(cudaMemcpy(h_hit_prim_id.data(), d_hit_prim_id, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy hit_prim");
  check_cuda(cudaMemcpy(h_hit_flag.data(), d_hit_flag, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy hit_flag");

  auto org_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.origins);
  auto dir_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rays.directions);
  auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.positions);
  auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.indices);

  Kokkos::View<Vec3 *, Kokkos::HostSpace> nrm_h;
  if (m_mesh.has_normals())
    nrm_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.normals);
  Kokkos::View<Vec2 *, Kokkos::HostSpace> tc_h;
  if (m_mesh.has_texcoords())
    tc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.texcoords);
  Kokkos::View<u32 *, Kokkos::HostSpace> mat_h;
  if (m_mesh.has_material_ids())
    mat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.material_ids);
  Kokkos::View<Vec3 *, Kokkos::HostSpace> vc_h;
  if (m_mesh.has_vertex_colors())
    vc_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.vertex_colors);

  auto hit_view = Kokkos::create_mirror_view(hits.hits);

  for (u32 i = 0; i < n; ++i) {
    HitResult hr{};
    hr.hit = h_hit_flag[i] != 0;
    if (hr.hit) {
      hr.t = h_hit_t[i];
      hr.position = org_h(i) + dir_h(i) * hr.t;
      hr.prim_id = h_hit_prim_id[i];
      hr.geom_id = 0;
      hr.inst_id = 0;

          const f32 bu = h_bary_u[i];
      const f32 bv = h_bary_v[i];
      const f32 bw = 1.f - bu - bv;

      const u32 pid = hr.prim_id;
      const u32 i0 = idx_h(pid * 3 + 0);
      const u32 i1 = idx_h(pid * 3 + 1);
      const u32 i2 = idx_h(pid * 3 + 2);

      const Vec3 p0 = pos_h(i0), p1 = pos_h(i1), p2 = pos_h(i2);
      Vec3 ng = cross(p1 - p0, p2 - p0);
      const f32 ng_len = length(ng);
      hr.normal = ng_len > 0.f ? ng * (1.f / ng_len) : Vec3{0, 1, 0};

      if (m_mesh.has_normals()) {
        Vec3 sn = nrm_h(i0) * bw + nrm_h(i1) * bu + nrm_h(i2) * bv;
        const f32 sn_len = length(sn);
        hr.shading_normal = sn_len > 0.f ? sn * (1.f / sn_len) : hr.normal;
      } else {
        hr.shading_normal = hr.normal;
      }

      if (m_mesh.has_texcoords()) {
        const Vec2 t0 = tc_h(i0), t1 = tc_h(i1), t2 = tc_h(i2);
        hr.uv = {t0.x * bw + t1.x * bu + t2.x * bv, t0.y * bw + t1.y * bu + t2.y * bv};
      } else {
        hr.uv = {bu, bv};
      }

      hr.material_id = m_mesh.has_material_ids() ? mat_h(pid) : 0u;

      if (m_mesh.has_vertex_colors() && vc_h(i0).x >= 0.f) {
        hr.interpolated_color = vc_h(i0) * bw + vc_h(i1) * bu + vc_h(i2) * bv;
        hr.has_interpolated_color = true;
      }
    }
    hit_view(i) = hr;
  }

  Kokkos::deep_copy(hits.hits, hit_view);

  cudaFree(d_hit_t);
  cudaFree(d_hit_prim_id);
  cudaFree(d_hit_flag);
  cudaFree(d_hit_bary_u);
  cudaFree(d_hit_bary_v);
  cudaFree(reinterpret_cast<void *>(d_params));
}

void OptixBackend::trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded)
{
  const u32 n = rays.count;

  unsigned int *d_occluded = nullptr;
  check_cuda(cudaMalloc(&d_occluded, n * sizeof(unsigned int)), "malloc occluded");
  check_cuda(cudaMemset(d_occluded, 0, n * sizeof(unsigned int)), "memset occluded");

  float3 *d_origins = reinterpret_cast<float3 *>(rays.origins.data());
  float3 *d_directions = reinterpret_cast<float3 *>(rays.directions.data());

  Params params{};
  params.handle = m_gas_handle;
  params.origins = d_origins;
  params.directions = d_directions;
  params.tmin = rays.tmin.data();
  params.tmax = rays.tmax.data();
  params.hit_t = nullptr;
  params.hit_prim_id = nullptr;
  params.hit_flag = nullptr;
  params.hit_bary_u = nullptr;
  params.hit_bary_v = nullptr;
  params.occluded = d_occluded;
  params.count = n;
  params.mode = 1;

  CUdeviceptr d_params = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)), "malloc params");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice), "copy params");

  check_optix(optixLaunch(m_pipeline, 0, d_params, sizeof(Params), &m_sbt, n, 1, 1), "optixLaunch occluded");
  check_cuda(cudaDeviceSynchronize(), "sync after occluded");

  auto occ_h = Kokkos::create_mirror_view(occluded);
  std::vector<unsigned int> h_occ(n);
  check_cuda(cudaMemcpy(h_occ.data(), d_occluded, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy occ");
  for (u32 i = 0; i < n; ++i)
    occ_h(i) = h_occ[i];
  Kokkos::deep_copy(occluded, occ_h);

  cudaFree(d_occluded);
  cudaFree(reinterpret_cast<void *>(d_params));
}

}

#endif
