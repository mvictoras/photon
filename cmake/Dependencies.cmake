include_guard(GLOBAL)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

option(OPENCODE_ENABLE_CUDA "Enable Kokkos CUDA backend" OFF)
option(OPENCODE_ENABLE_HIP "Enable Kokkos HIP backend" OFF)
option(OPENCODE_ENABLE_SYCL "Enable Kokkos SYCL backend" OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)

if(OPENCODE_ENABLE_CUDA)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
if(OPENCODE_ENABLE_HIP)
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
endif()
if(OPENCODE_ENABLE_SYCL)
  set(Kokkos_ENABLE_SYCL ON CACHE BOOL "" FORCE)
endif()

set(Kokkos_ENABLE_MDSPAN OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(kokkos)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_VIEWER OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE) # dependency testing
set(BUILD_CTS OFF CACHE BOOL "" FORCE)
set(BUILD_CAT OFF CACHE BOOL "" FORCE)
set(BUILD_HELIUM_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  anari
  GIT_REPOSITORY https://github.com/KhronosGroup/ANARI-SDK.git
  GIT_TAG v0.15.0
  PATCH_COMMAND find . -name CMakeLists.txt -exec sed -i.bak "s/CMAKE_SOURCE_DIR/PROJECT_SOURCE_DIR/g" {} +
)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE) # anari-sdk testing (keep OFF)
set(BUILD_HELIDE_DEVICE OFF CACHE BOOL "" FORCE)
set(BUILD_DEBUG_DEVICE OFF CACHE BOOL "" FORCE)
set(BUILD_SINK_DEVICE OFF CACHE BOOL "" FORCE)
set(BUILD_CTS OFF CACHE BOOL "" FORCE)
set(BUILD_CAT OFF CACHE BOOL "" FORCE)
set(INSTALL_CODE_GEN_SCRIPTS OFF CACHE BOOL "" FORCE)
set(INSTALL_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(ENABLE_RENDER_TESTS OFF CACHE BOOL "" FORCE)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE) # ensure dependency tests disabled

FetchContent_MakeAvailable(anari)

if(NOT TARGET anari)
  message(FATAL_ERROR "ANARI target not found (expected 'anari')")
endif()

option(PHOTON_ENABLE_EMBREE "Enable Embree ray tracing backend" ON)
if(PHOTON_ENABLE_EMBREE)
  find_package(embree 4 QUIET)
  if(embree_FOUND)
    message(STATUS "Embree found: ${embree_DIR}")
    target_link_libraries(anari_library_photon PUBLIC embree)
    target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_EMBREE)
  else()
    message(STATUS "Embree not found — Embree backend disabled")
  endif()
endif()

option(PHOTON_ENABLE_OPTIX "Enable OptiX ray tracing backend" OFF)
if(PHOTON_ENABLE_OPTIX)
  find_package(CUDAToolkit REQUIRED)
  find_path(OptiX_INCLUDE_DIR optix.h
    PATHS $ENV{OptiX_INSTALL_DIR}/include /usr/local/include
    DOC "OptiX include directory")
  if(OptiX_INCLUDE_DIR)
    message(STATUS "OptiX found: ${OptiX_INCLUDE_DIR}")
    target_include_directories(anari_library_photon PUBLIC ${OptiX_INCLUDE_DIR})
    target_link_libraries(anari_library_photon PUBLIC CUDA::cudart)
    target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OPTIX)
  else()
    message(FATAL_ERROR "OptiX headers not found. Set OptiX_INSTALL_DIR.")
  endif()
endif()

option(PHOTON_ENABLE_HIPRT "Enable HIP RT ray tracing backend" OFF)
if(PHOTON_ENABLE_HIPRT)
  find_package(hiprt REQUIRED)
  target_link_libraries(anari_library_photon PUBLIC hiprt::hiprt)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_HIPRT)
endif()

option(PHOTON_ENABLE_OIDN "Enable Intel OIDN denoiser" ON)
if(PHOTON_ENABLE_OIDN)
  find_package(OpenImageDenoise 2 QUIET)
  if(OpenImageDenoise_FOUND)
    target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OIDN)
    target_link_libraries(anari_library_photon PUBLIC OpenImageDenoise)
  else()
    message(STATUS "Intel OIDN not found — denoiser disabled")
  endif()
endif()
