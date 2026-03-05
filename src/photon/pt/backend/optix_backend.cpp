#ifdef PHOTON_HAS_OPTIX

#include "photon/pt/backend/optix_backend.h"

#include "photon/pt/scene.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace photon::pt {
namespace {

constexpr unsigned OPTIX_SBT_RECORD_ALIGNMENT = OPTIX_SBT_RECORD_ALIGNMENT;

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecordHeader
{
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct Params
{
  OptixTraversableHandle handle;
  const float3 *origins;
  const float3 *directions;
  const float *tmin;
  const float *tmax;
  HitResult *hits;
  unsigned *occluded;
  unsigned count;
  unsigned mode;
};

static void optix_log_callback(unsigned int level, const char *tag, const char *message, void *)
{
  (void)level;
  std::fprintf(stderr, "[photon][optix][%s] %s\n", tag ? tag : "", message ? message : "");
}

inline void check_optix(const OptixResult r, const char *what)
{
  if (r != OPTIX_SUCCESS) {
    throw std::runtime_error(std::string("OptiX error ") + what);
  }
}

inline void check_cuda(const cudaError_t e, const char *what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error ") + what + ": " + cudaGetErrorString(e));
  }
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
  pipeline_compile_options.numPayloadValues = 4;
  pipeline_compile_options.numAttributeValues = 2;
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  OptixModuleCompileOptions module_compile_options{};
  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  extern const char photon_optix_ptx[];
  extern const size_t photon_optix_ptx_size;

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
  hit_desc.hitgroup.moduleAH = m_module;
  hit_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";

  OptixProgramGroup raygen_pg = nullptr;
  OptixProgramGroup miss_pg = nullptr;
  OptixProgramGroup hit_pg = nullptr;

  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &raygen_desc, 1, &pg_options, log, &log_size, &raygen_pg),
              "optixProgramGroupCreate raygen");
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &miss_desc, 1, &pg_options, log, &log_size, &miss_pg),
              "optixProgramGroupCreate miss");
  log_size = sizeof(log);
  check_optix(optixProgramGroupCreate(m_context, &hit_desc, 1, &pg_options, log, &log_size, &hit_pg),
              "optixProgramGroupCreate hit");

  std::vector<OptixProgramGroup> groups{raygen_pg, miss_pg, hit_pg};

  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;
  link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  log_size = sizeof(log);
  check_optix(optixPipelineCreate(
                  m_context,
                  &pipeline_compile_options,
                  &link_options,
                  groups.data(),
                  static_cast<unsigned>(groups.size()),
                  log,
                  &log_size,
                  &m_pipeline),
              "optixPipelineCreate");

  OptixStackSizes ss{};
  for (auto pg : groups) {
    optixUtilAccumulateStackSizes(pg, &ss);
  }

  uint32_t direct_callable_stack_size_from_traversal = 0;
  uint32_t direct_callable_stack_size_from_state = 0;
  uint32_t continuation_stack_size = 0;
  optixUtilComputeStackSizes(&ss, link_options.maxTraceDepth, 0, 0, &direct_callable_stack_size_from_traversal,
                             &direct_callable_stack_size_from_state, &continuation_stack_size);
  check_optix(optixPipelineSetStackSize(m_pipeline,
                                       direct_callable_stack_size_from_traversal,
                                       direct_callable_stack_size_from_state,
                                       continuation_stack_size,
                                       1),
              "optixPipelineSetStackSize");

  optixProgramGroupDestroy(raygen_pg);
  optixProgramGroupDestroy(miss_pg);
  optixProgramGroupDestroy(hit_pg);
}

void OptixBackend::create_sbt()
{
  if (m_sbt.raygenRecord)
    return;

  OptixProgramGroupDesc dummy{};
  (void)dummy;

  struct EmptyData
  {
  };

  SbtRecord<EmptyData> rg{};
  SbtRecord<EmptyData> ms{};
  SbtRecord<EmptyData> hg{};

  CUdeviceptr d_rg = 0;
  CUdeviceptr d_ms = 0;
  CUdeviceptr d_hg = 0;

  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rg)), "cudaMalloc sbt rg");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ms), sizeof(ms)), "cudaMalloc sbt ms");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_hg), sizeof(hg)), "cudaMalloc sbt hg");

  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rg, sizeof(rg), cudaMemcpyHostToDevice), "cudaMemcpy rg");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_ms), &ms, sizeof(ms), cudaMemcpyHostToDevice), "cudaMemcpy ms");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_hg), &hg, sizeof(hg), cudaMemcpyHostToDevice), "cudaMemcpy hg");

  m_sbt.raygenRecord = d_rg;
  m_sbt.missRecordBase = d_ms;
  m_sbt.missRecordStrideInBytes = sizeof(ms);
  m_sbt.missRecordCount = 1;
  m_sbt.hitgroupRecordBase = d_hg;
  m_sbt.hitgroupRecordStrideInBytes = sizeof(hg);
  m_sbt.hitgroupRecordCount = 1;
}

void OptixBackend::build_accel(const Scene &scene)
{
  m_mesh = scene.mesh;

  auto pos_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.positions);
  auto idx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m_mesh.indices);

  const size_t vertex_count = pos_h.extent(0);
  const size_t tri_count = idx_h.extent(0) / 3;

  std::vector<float3> vertices(vertex_count);
  for (size_t i = 0; i < vertex_count; ++i) {
    vertices[i] = make_float3(pos_h(i).x, pos_h(i).y, pos_h(i).z);
  }

  std::vector<uint3> indices(tri_count);
  for (size_t i = 0; i < tri_count; ++i) {
    indices[i] = make_uint3(idx_h(i * 3 + 0), idx_h(i * 3 + 1), idx_h(i * 3 + 2));
  }

  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_indices = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices.size() * sizeof(float3)), "cudaMalloc vertices");
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_indices), indices.size() * sizeof(uint3)), "cudaMalloc indices");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_vertices), vertices.data(), vertices.size() * sizeof(float3),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy vertices");
  check_cuda(cudaMemcpy(reinterpret_cast<void *>(d_indices), indices.data(), indices.size() * sizeof(uint3),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy indices");

  OptixBuildInput build_input{};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  OptixBuildInputTriangleArray &tri = build_input.triangleArray;
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
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes buffer_sizes{};
  check_optix(optixAccelComputeMemoryUsage(m_context, &accel_options, &build_input, 1, &buffer_sizes),
              "optixAccelComputeMemoryUsage");

  CUdeviceptr d_temp = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_temp), buffer_sizes.tempSizeInBytes), "cudaMalloc temp");

  if (m_gas_buffer) {
    cudaFree(reinterpret_cast<void *>(m_gas_buffer));
    m_gas_buffer = 0;
  }
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&m_gas_buffer), buffer_sizes.outputSizeInBytes), "cudaMalloc gas");

  OptixAccelEmitDesc emit_desc{};
  emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  CUdeviceptr d_compacted_size = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_compacted_size), sizeof(size_t)), "cudaMalloc compacted size");
  emit_desc.result = d_compacted_size;

  check_optix(optixAccelBuild(m_context,
                             0,
                             &accel_options,
                             &build_input,
                             1,
                             d_temp,
                             buffer_sizes.tempSizeInBytes,
                             m_gas_buffer,
                             buffer_sizes.outputSizeInBytes,
                             &m_gas_handle,
                             &emit_desc,
                             1),
              "optixAccelBuild");

  check_cuda(cudaFree(reinterpret_cast<void *>(d_temp)), "cudaFree temp");

  size_t compacted_size = 0;
  check_cuda(cudaMemcpy(&compacted_size, reinterpret_cast<void *>(d_compacted_size), sizeof(size_t), cudaMemcpyDeviceToHost),
             "cudaMemcpy compacted size");
  check_cuda(cudaFree(reinterpret_cast<void *>(d_compacted_size)), "cudaFree compacted size");

  if (compacted_size > 0 && compacted_size < buffer_sizes.outputSizeInBytes) {
    CUdeviceptr d_compacted = 0;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_compacted), compacted_size), "cudaMalloc gas compacted");
    check_optix(optixAccelCompact(m_context, 0, m_gas_handle, d_compacted, compacted_size, &m_gas_handle),
                "optixAccelCompact");
    check_cuda(cudaFree(reinterpret_cast<void *>(m_gas_buffer)), "cudaFree gas");
    m_gas_buffer = d_compacted;
  }

  check_cuda(cudaFree(reinterpret_cast<void *>(d_vertices)), "cudaFree vertices");
  check_cuda(cudaFree(reinterpret_cast<void *>(d_indices)), "cudaFree indices");

  create_pipeline();
  create_sbt();
}

void OptixBackend::trace_closest(const RayBatch &, HitBatch &)
{
  throw std::runtime_error("OptixBackend::trace_closest not implemented: requires device programs + SBT records");
}

void OptixBackend::trace_occluded(const RayBatch &, Kokkos::View<u32 *>)
{
  throw std::runtime_error("OptixBackend::trace_occluded not implemented: requires device programs + SBT records");
}

}

#endif
