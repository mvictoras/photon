include_guard(GLOBAL)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

option(OPENCODE_ENABLE_CUDA "Enable Kokkos CUDA backend" OFF)
option(OPENCODE_ENABLE_HIP "Enable Kokkos HIP backend" OFF)
option(OPENCODE_ENABLE_SYCL "Enable Kokkos SYCL backend" OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

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
