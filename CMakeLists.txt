cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0079 NEW)

# Kokkos CUDA uses COMPILE_AS_CMAKE_LANGUAGE mode: CMake's native CUDA language
# support compiles .cpp files as CUDA via nvcc with MSVC as the host compiler.
# Only photon source files are compiled as CUDA — ANARI SDK files are NOT
# (CUDA's float4 conflicts with linalg::float4).

option(OPENCODE_ENABLE_CUDA "Enable Kokkos CUDA backend for GPU shading" ON)

if(OPENCODE_ENABLE_CUDA)
  project(opencode_pathtracer LANGUAGES CXX CUDA)
  # Ensure CUDA standard matches CXX
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  project(opencode_pathtracer LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

set(BUILD_TESTING ON) # this project

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectOptions)
include(Dependencies)

add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

add_subdirectory(app)
add_subdirectory(tools)
