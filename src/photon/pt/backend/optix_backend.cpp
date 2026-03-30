#ifdef PHOTON_HAS_OPTIX

#include "photon/pt/backend/optix_backend.h"
#include "photon/pt/scene.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace photon::pt {

// True when Kokkos default execution space is NOT CUDA.
// In that case, Kokkos::View .data() returns host pointers which cannot
// be dereferenced from OptiX device code; we must stage via cudaMemcpy.
static constexpr bool needs_staging =
#ifdef KOKKOS_ENABLE_CUDA
    !std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value;
#else
    true;  // No CUDA backend compiled into Kokkos -> always stage
#endif

namespace {

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

} // anonymous namespace

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

// Upload host data to a new CUDA device allocation. Returns device pointer.
static void *cuda_upload(const void *host_ptr, size_t bytes)
{
  if (!host_ptr || bytes == 0) return nullptr;
  void *d = nullptr;
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&d, bytes)), "staging alloc");
  check_cuda(cudaMemcpy(d, host_ptr, bytes, cudaMemcpyHostToDevice), "staging upload");
  return d;
}

// Allocate device memory and return pointer (contents undefined).
static void *cuda_alloc(size_t bytes)
{
  if (bytes == 0) return nullptr;
  void *d = nullptr;
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&d, bytes)), "staging alloc out");
  return d;
}

// Download device data to host.
static void cuda_download(void *host_ptr, const void *dev_ptr, size_t bytes)
{
  if (!host_ptr || !dev_ptr || bytes == 0) return;
  check_cuda(cudaMemcpy(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost), "staging download");
}

OptixBackend::OptixBackend()
{
  check_cuda(cudaFree(nullptr), "init cuda");
  check_cuda(cudaStreamCreate(&m_stream), "streamCreate");
  CUcontext cu_ctx = 0;
  OptixDeviceContextOptions options{};
  options.logCallbackFunction = optix_log_callback;
  options.logCallbackLevel = 4;
  check_optix(optixInit(), "optixInit");
  check_optix(optixDeviceContextCreate(cu_ctx, &options, &m_context), "optixDeviceContextCreate");
}

void OptixBackend::free_device_mesh_buffers()
{
  if (m_d_mesh_positions)    { cudaFree(m_d_mesh_positions);    m_d_mesh_positions = nullptr; }
  if (m_d_mesh_indices)      { cudaFree(m_d_mesh_indices);      m_d_mesh_indices = nullptr; }
  if (m_d_mesh_normals)      { cudaFree(m_d_mesh_normals);      m_d_mesh_normals = nullptr; }
  if (m_d_mesh_texcoords)    { cudaFree(m_d_mesh_texcoords);    m_d_mesh_texcoords = nullptr; }
  if (m_d_mesh_material_ids) { cudaFree(m_d_mesh_material_ids); m_d_mesh_material_ids = nullptr; }
  if (m_d_mesh_vertex_colors){ cudaFree(m_d_mesh_vertex_colors);m_d_mesh_vertex_colors = nullptr; }
}

void OptixBackend::free_staging_buffers()
{
  if (m_d_ray_origins)    { cudaFree(m_d_ray_origins);    m_d_ray_origins = nullptr; }
  if (m_d_ray_directions) { cudaFree(m_d_ray_directions); m_d_ray_directions = nullptr; }
  if (m_d_ray_tmin)       { cudaFree(m_d_ray_tmin);       m_d_ray_tmin = nullptr; }
  if (m_d_ray_tmax)       { cudaFree(m_d_ray_tmax);       m_d_ray_tmax = nullptr; }
  if (m_d_hit_results)    { cudaFree(m_d_hit_results);    m_d_hit_results = nullptr; }
  if (m_d_occluded)       { cudaFree(m_d_occluded);       m_d_occluded = nullptr; }
  m_staging_capacity = 0;
}

void OptixBackend::ensure_staging_buffers(u32 count)
{
  if (count <= m_staging_capacity) return;

  free_staging_buffers();
  m_staging_capacity = count;

  check_cuda(static_cast<cudaError_t>(cudaMalloc(&m_d_ray_origins,    count * sizeof(Vec3))),      "staging origins");
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&m_d_ray_directions, count * sizeof(Vec3))),      "staging dirs");
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&m_d_ray_tmin,       count * sizeof(f32))),       "staging tmin");
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&m_d_ray_tmax,       count * sizeof(f32))),       "staging tmax");
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&m_d_hit_results,    count * sizeof(HitResult))), "staging hits");
  check_cuda(static_cast<cudaError_t>(cudaMalloc(&m_d_occluded,       count * sizeof(u32))),       "staging occluded");
}

OptixBackend::~OptixBackend()
{
  free_staging_buffers();
  free_device_mesh_buffers();
  if (m_pipeline) optixPipelineDestroy(m_pipeline);
  if (m_module) optixModuleDestroy(m_module);
  if (m_gas_buffer) cudaFree(reinterpret_cast<void *>(m_gas_buffer));
  if (m_d_params) cudaFree(reinterpret_cast<void *>(m_d_params));
  if (m_context) optixDeviceContextDestroy(m_context);
  if (m_stream) cudaStreamDestroy(m_stream);
}

extern const char photon_optix_ptx[];
extern const unsigned long long photon_optix_ptx_size;

void OptixBackend::create_pipeline()
{
  if (m_pipeline) return;

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

  char log[2048]{};
  size_t log_size = sizeof(log);

  check_optix(optixModuleCreate(m_context, &module_compile_options, &pipeline_compile_options,
      photon_optix_ptx, photon_optix_ptx_size, log, &log_size, &m_module), "moduleCreate");

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

  OptixProgramGroup raygen_pg = nullptr, miss_pg = nullptr, hit_pg = nullptr;
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &raygen_desc, 1, &pg_options, log, &log_size, &raygen_pg), "raygen pg");
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &miss_desc, 1, &pg_options, log, &log_size, &miss_pg), "miss pg");
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &hit_desc, 1, &pg_options, log, &log_size, &hit_pg), "hit pg");

  OptixProgramGroup groups[] = {raygen_pg, miss_pg, hit_pg};
  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;

  log_size = sizeof(log);
  check_optix(optixPipelineCreate(m_context, &pipeline_compile_options, &link_options,
      groups, 3, log, &log_size, &m_pipeline), "pipelineCreate");
  check_optix(optixPipelineSetStackSize(m_pipeline, 2048, 2048, 2048, 1), "setStackSize");

  SbtRecord<EmptyData> rg_record{}, ms_record{}, hg_record{};
  optixSbtRecordPackHeader(raygen_pg, &rg_record);
  optixSbtRecordPackHeader(miss_pg, &ms_record);
  optixSbtRecordPackHeader(hit_pg, &hg_record);

  CUdeviceptr d_rg = 0, d_ms = 0, d_hg = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rg_record)), "sbt rg");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ms), sizeof(ms_record)), "sbt ms");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_hg), sizeof(hg_record)), "sbt hg");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rg_record, sizeof(rg_record), cudaMemcpyHostToDevice), "sbt rg cp");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_ms), &ms_record, sizeof(ms_record), cudaMemcpyHostToDevice), "sbt ms cp");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_hg), &hg_record, sizeof(hg_record), cudaMemcpyHostToDevice), "sbt hg cp");

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

  check_cuda(cudaMalloc(reinterpret_cast<void **>(&m_d_params), sizeof(Params)), "params alloc");
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
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices.size() * sizeof(float3)), "v");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_indices), indices.size() * sizeof(uint3)), "i");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_vertices), vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice), "v cp");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_indices), indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice), "i cp");

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
  check_optix(optixAccelComputeMemoryUsage(m_context, &accel_options, &build_input, 1, &buffer_sizes), "accel mem");

  CUdeviceptr d_temp = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_temp), buffer_sizes.tempSizeInBytes), "temp");

  if (m_gas_buffer) { cudaFree(reinterpret_cast<void *>(m_gas_buffer)); m_gas_buffer = 0; }
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&m_gas_buffer), buffer_sizes.outputSizeInBytes), "gas");

  CUdeviceptr d_compacted_size = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_compacted_size), sizeof(size_t)), "compact sz");
  OptixAccelEmitDesc emit_desc{};
  emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_desc.result = d_compacted_size;

  check_optix(optixAccelBuild(m_context, 0, &accel_options, &build_input, 1,
      d_temp, buffer_sizes.tempSizeInBytes, m_gas_buffer, buffer_sizes.outputSizeInBytes,
      &m_gas_handle, &emit_desc, 1), "accelBuild");

  cudaFree(reinterpret_cast<void *>(d_temp));

  size_t compacted_size = 0;
  cudaMemcpy(&compacted_size, reinterpret_cast<void *>(d_compacted_size), sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaFree(reinterpret_cast<void *>(d_compacted_size));

  if (compacted_size > 0 && compacted_size < buffer_sizes.outputSizeInBytes) {
    CUdeviceptr d_compacted = 0;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_compacted), compacted_size), "compact");
    check_optix(optixAccelCompact(m_context, 0, m_gas_handle, d_compacted, compacted_size, &m_gas_handle), "compact");
    cudaFree(reinterpret_cast<void *>(m_gas_buffer));
    m_gas_buffer = d_compacted;
  }

  cudaFree(reinterpret_cast<void *>(d_vertices));
  cudaFree(reinterpret_cast<void *>(d_indices));

  std::fprintf(stderr, "[photon] OptiX GAS built: %zu tris, %.1f KB\n",
      tri_count, float(compacted_size > 0 ? compacted_size : buffer_sizes.outputSizeInBytes) / 1024.f);

  create_pipeline();

  // When Kokkos uses a non-CUDA backend (Serial, OpenMP, etc.), the mesh
  // views live in host memory.  Upload them to persistent device buffers
  // so the OptiX closest-hit shader can access mesh attributes.
  if (needs_staging) {
    free_device_mesh_buffers();
    m_d_mesh_positions = cuda_upload(m_mesh.positions.data(),
        m_mesh.positions.extent(0) * sizeof(Vec3));
    m_d_mesh_indices = cuda_upload(m_mesh.indices.data(),
        m_mesh.indices.extent(0) * sizeof(u32));
    if (m_mesh.has_normals())
      m_d_mesh_normals = cuda_upload(m_mesh.normals.data(),
          m_mesh.normals.extent(0) * sizeof(Vec3));
    if (m_mesh.has_texcoords())
      m_d_mesh_texcoords = cuda_upload(m_mesh.texcoords.data(),
          m_mesh.texcoords.extent(0) * sizeof(Vec2));
    if (m_mesh.has_material_ids())
      m_d_mesh_material_ids = cuda_upload(m_mesh.material_ids.data(),
          m_mesh.material_ids.extent(0) * sizeof(u32));
    if (m_mesh.has_vertex_colors())
      m_d_mesh_vertex_colors = cuda_upload(m_mesh.vertex_colors.data(),
          m_mesh.vertex_colors.extent(0) * sizeof(Vec3));
  }
}

void OptixBackend::trace_closest(const RayBatch &rays, HitBatch &hits)
{
  const u32 n = rays.count;

  if (needs_staging) {
    ensure_staging_buffers(n);
    check_cuda(cudaMemcpy(m_d_ray_origins,    rays.origins.data(),    n * sizeof(Vec3), cudaMemcpyHostToDevice), "upload origins");
    check_cuda(cudaMemcpy(m_d_ray_directions, rays.directions.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice), "upload dirs");
    check_cuda(cudaMemcpy(m_d_ray_tmin,       rays.tmin.data(),       n * sizeof(f32),  cudaMemcpyHostToDevice), "upload tmin");
    check_cuda(cudaMemcpy(m_d_ray_tmax,       rays.tmax.data(),       n * sizeof(f32),  cudaMemcpyHostToDevice), "upload tmax");
  }

  Params params{};
  params.handle = m_gas_handle;
  params.origins    = reinterpret_cast<float3 *>(needs_staging ? m_d_ray_origins    : rays.origins.data());
  params.directions = reinterpret_cast<float3 *>(needs_staging ? m_d_ray_directions : rays.directions.data());
  params.tmin       = reinterpret_cast<float *> (needs_staging ? m_d_ray_tmin       : rays.tmin.data());
  params.tmax       = reinterpret_cast<float *> (needs_staging ? m_d_ray_tmax       : rays.tmax.data());
  params.hit_results = needs_staging ? m_d_hit_results : hits.hits.data();
  params.occluded = nullptr;

  if (needs_staging) {
    params.mesh_positions     = reinterpret_cast<float3 *>(m_d_mesh_positions);
    params.mesh_indices       = reinterpret_cast<unsigned int *>(m_d_mesh_indices);
    params.mesh_normals       = reinterpret_cast<float3 *>(m_d_mesh_normals);
    params.mesh_texcoords     = reinterpret_cast<float2 *>(m_d_mesh_texcoords);
    params.mesh_material_ids  = reinterpret_cast<unsigned int *>(m_d_mesh_material_ids);
    params.mesh_vertex_colors = reinterpret_cast<float3 *>(m_d_mesh_vertex_colors);
  } else {
    params.mesh_positions     = reinterpret_cast<float3 *>(m_mesh.positions.data());
    params.mesh_indices       = reinterpret_cast<unsigned int *>(m_mesh.indices.data());
    params.mesh_normals       = m_mesh.has_normals()       ? reinterpret_cast<float3 *>(m_mesh.normals.data()) : nullptr;
    params.mesh_texcoords     = m_mesh.has_texcoords()     ? reinterpret_cast<float2 *>(m_mesh.texcoords.data()) : nullptr;
    params.mesh_material_ids  = m_mesh.has_material_ids()  ? reinterpret_cast<unsigned int *>(m_mesh.material_ids.data()) : nullptr;
    params.mesh_vertex_colors = m_mesh.has_vertex_colors() ? reinterpret_cast<float3 *>(m_mesh.vertex_colors.data()) : nullptr;
  }

  params.has_normals       = m_mesh.has_normals()       ? 1 : 0;
  params.has_texcoords     = m_mesh.has_texcoords()     ? 1 : 0;
  params.has_material_ids  = m_mesh.has_material_ids()  ? 1 : 0;
  params.has_vertex_colors = m_mesh.has_vertex_colors() ? 1 : 0;

  params.count = n;
  params.mode = 0;

  check_cuda(cudaMemcpy(reinterpret_cast<void *>(m_d_params), &params, sizeof(Params), cudaMemcpyHostToDevice), "params cp");
  check_optix(optixLaunch(m_pipeline, m_stream, m_d_params, sizeof(Params), &m_sbt, n, 1, 1), "launch closest");
  check_cuda(cudaStreamSynchronize(m_stream), "sync closest");

  if (needs_staging) {
    check_cuda(cudaMemcpy(hits.hits.data(), m_d_hit_results, n * sizeof(HitResult), cudaMemcpyDeviceToHost), "download hits");
  }
}

void OptixBackend::trace_occluded(const RayBatch &rays, Kokkos::View<u32 *> occluded)
{
  const u32 n = rays.count;

  if (needs_staging) {
    ensure_staging_buffers(n);
    check_cuda(cudaMemcpy(m_d_ray_origins,    rays.origins.data(),    n * sizeof(Vec3), cudaMemcpyHostToDevice), "upload origins");
    check_cuda(cudaMemcpy(m_d_ray_directions, rays.directions.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice), "upload dirs");
    check_cuda(cudaMemcpy(m_d_ray_tmin,       rays.tmin.data(),       n * sizeof(f32),  cudaMemcpyHostToDevice), "upload tmin");
    check_cuda(cudaMemcpy(m_d_ray_tmax,       rays.tmax.data(),       n * sizeof(f32),  cudaMemcpyHostToDevice), "upload tmax");
  }

  Params params{};
  params.handle = m_gas_handle;
  params.origins    = reinterpret_cast<float3 *>(needs_staging ? m_d_ray_origins    : rays.origins.data());
  params.directions = reinterpret_cast<float3 *>(needs_staging ? m_d_ray_directions : rays.directions.data());
  params.tmin       = reinterpret_cast<float *> (needs_staging ? m_d_ray_tmin       : rays.tmin.data());
  params.tmax       = reinterpret_cast<float *> (needs_staging ? m_d_ray_tmax       : rays.tmax.data());
  params.hit_results = nullptr;
  params.occluded   = reinterpret_cast<unsigned int *>(needs_staging ? m_d_occluded : occluded.data());
  params.count = n;
  params.mode = 1;

  check_cuda(cudaMemcpy(reinterpret_cast<void *>(m_d_params), &params, sizeof(Params), cudaMemcpyHostToDevice), "params cp");
  check_optix(optixLaunch(m_pipeline, m_stream, m_d_params, sizeof(Params), &m_sbt, n, 1, 1), "launch occluded");
  check_cuda(cudaStreamSynchronize(m_stream), "sync occluded");

  if (needs_staging) {
    check_cuda(cudaMemcpy(occluded.data(), m_d_occluded, n * sizeof(u32), cudaMemcpyDeviceToHost), "download occluded");
  }
}

}

#endif
