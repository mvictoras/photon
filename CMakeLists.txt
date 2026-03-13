cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0079 NEW)

# When CUDA is enabled, Kokkos needs nvcc_wrapper as the CXX compiler globally.
# However, this conflicts with the ANARI SDK (CUDA's float4 clashes with linalg::float4).
# Instead, we build WITHOUT nvcc_wrapper globally and use kokkos_compilation() per-target.
# This requires a two-phase approach: first configure Kokkos without CUDA to get the
# cmake functions, then apply CUDA compilation rules to our targets only.

project(opencode_pathtracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
