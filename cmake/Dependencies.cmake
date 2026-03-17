include_guard(GLOBAL)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

option(OPENCODE_ENABLE_CUDA "Enable Kokkos CUDA backend" ON)
option(OPENCODE_ENABLE_HIP "Enable Kokkos HIP backend" OFF)
option(OPENCODE_ENABLE_SYCL "Enable Kokkos SYCL backend" OFF)

# --------------------------------------------------------------------------
# Kokkos: prefer a pre-installed build (required for CUDA to avoid
# nvcc_wrapper compiling the ANARI SDK which conflicts with CUDA's float4).
# Fall back to FetchContent if no installed Kokkos is found.
# --------------------------------------------------------------------------
find_package(Kokkos 5.0 QUIET)
if(Kokkos_FOUND)
  message(STATUS "Using pre-installed Kokkos: ${Kokkos_DIR}")
else()
  message(STATUS "Pre-installed Kokkos not found — fetching via FetchContent (CUDA disabled)")
  # FetchContent Kokkos cannot use CUDA without nvcc_wrapper as global compiler
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_MDSPAN OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 5.0.1
  )
  FetchContent_MakeAvailable(kokkos)
endif()

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
set(BUILD_HELIDE_DEVICE ON CACHE BOOL "" FORCE)
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
set(PHOTON_EMBREE_READY FALSE CACHE INTERNAL "")
if(PHOTON_ENABLE_EMBREE)
  find_package(embree 4 QUIET
    HINTS $ENV{EMBREE_DIR} $ENV{HOME}/src/embree-install
  )
  if(embree_FOUND)
    message(STATUS "Embree found: ${embree_DIR}")
    set(PHOTON_EMBREE_READY TRUE CACHE INTERNAL "")
  else()
    message(STATUS "Embree not found — Embree backend disabled. Set EMBREE_DIR or install Embree 4.")
  endif()
endif()

option(PHOTON_ENABLE_OPTIX "Enable OptiX ray tracing backend" OFF)
set(PHOTON_OPTIX_READY FALSE CACHE INTERNAL "")
if(PHOTON_ENABLE_OPTIX)
  find_package(CUDAToolkit REQUIRED)
  find_path(OptiX_INCLUDE_DIR optix.h
    PATHS $ENV{OptiX_INSTALL_DIR}/include
          $ENV{HOME}/src/optix-headers/include
          /usr/local/include
    DOC "OptiX include directory")
  if(OptiX_INCLUDE_DIR)
    message(STATUS "OptiX found: ${OptiX_INCLUDE_DIR}")
    set(PHOTON_OPTIX_INCLUDE_DIR "${OptiX_INCLUDE_DIR}" CACHE INTERNAL "")
    set(PHOTON_OPTIX_READY TRUE CACHE INTERNAL "")
  else()
    message(FATAL_ERROR "OptiX headers not found. Set OptiX_INSTALL_DIR.")
  endif()
endif()

option(PHOTON_ENABLE_HIPRT "Enable HIP RT ray tracing backend" OFF)
set(PHOTON_HIPRT_READY FALSE CACHE INTERNAL "")
if(PHOTON_ENABLE_HIPRT)
  find_package(hiprt REQUIRED)
  set(PHOTON_HIPRT_READY TRUE CACHE INTERNAL "")
endif()

option(PHOTON_ENABLE_OIDN "Enable Intel OIDN denoiser" ON)
set(PHOTON_OIDN_READY FALSE CACHE INTERNAL "")
if(PHOTON_ENABLE_OIDN)
  find_package(OpenImageDenoise 2 QUIET
    HINTS /opt/hfs21.0.596/toolkit/cmake)
  if(OpenImageDenoise_FOUND)
    set(PHOTON_OIDN_READY TRUE CACHE INTERNAL "")
    message(STATUS "OIDN found: ${OpenImageDenoise_DIR}")
  else()
    find_path(OIDN_INCLUDE_DIR OpenImageDenoise/oidn.hpp
      HINTS /opt/hfs21.0.596/toolkit/include $ENV{OIDN_DIR}/include)
    find_library(OIDN_LIBRARY NAMES OpenImageDenoise
      HINTS /opt/hfs21.0.596/dsolib $ENV{OIDN_DIR}/lib)
    if(OIDN_INCLUDE_DIR AND OIDN_LIBRARY)
      set(PHOTON_OIDN_READY TRUE CACHE INTERNAL "")
      set(PHOTON_OIDN_INCLUDE "${OIDN_INCLUDE_DIR}" CACHE INTERNAL "")
      set(PHOTON_OIDN_LIB "${OIDN_LIBRARY}" CACHE INTERNAL "")
      message(STATUS "OIDN found manually: ${OIDN_INCLUDE_DIR}")
    else()
      message(STATUS "Intel OIDN not found — denoiser disabled")
    endif()
  endif()
endif()

option(PHOTON_ENABLE_MPI "Enable MPI for data-parallel rendering" OFF)
set(PHOTON_MPI_READY FALSE CACHE INTERNAL "")
set(PHOTON_ICET_READY FALSE CACHE INTERNAL "")
if(PHOTON_ENABLE_MPI)
  find_package(MPI REQUIRED)
  if(MPI_CXX_FOUND)
    set(PHOTON_MPI_READY TRUE CACHE INTERNAL "")
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")

    set(ICET_USE_OPENGL OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      icet
      GIT_REPOSITORY https://gitlab.kitware.com/icet/icet.git
      GIT_TAG IceT-2.1.1
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(icet)
    set(PHOTON_ICET_READY TRUE CACHE INTERNAL "")
    message(STATUS "IceT fetched and built")
  endif()
endif()
