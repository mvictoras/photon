add_library(opencode_pathtracer STATIC)

opencode_enable_warnings(opencode_pathtracer)

target_link_libraries(opencode_pathtracer PUBLIC Kokkos::kokkos anari)

target_include_directories(opencode_pathtracer PUBLIC "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include")

target_include_directories(opencode_pathtracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(opencode_pathtracer PUBLIC cxx_std_23)

target_sources(opencode_pathtracer
  PRIVATE
    opencode/pathtracer/version.cpp
    opencode/anari/device.cpp
  PUBLIC
    opencode/pathtracer/version.h
    opencode/anari/device.h
)
