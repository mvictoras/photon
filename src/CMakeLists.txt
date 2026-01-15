add_library(anari_library_opencode_pathtracer SHARED)
set_target_properties(anari_library_opencode_pathtracer PROPERTIES OUTPUT_NAME "anari_library_opencode_pathtracer")

opencode_enable_warnings(anari_library_opencode_pathtracer)

target_link_libraries(anari_library_opencode_pathtracer PUBLIC Kokkos::kokkos anari)

target_include_directories(anari_library_opencode_pathtracer PUBLIC
  "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include"
  "${CMAKE_BINARY_DIR}/_deps/anari-src/src"
)

target_include_directories(anari_library_opencode_pathtracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(anari_library_opencode_pathtracer PUBLIC cxx_std_23)

target_sources(anari_library_opencode_pathtracer
  PRIVATE
    opencode/pathtracer/version.cpp
    opencode/anari/device.cpp
    opencode/anari/library.cpp
    opencode/anari/entrypoint.cpp
    opencode/pt/pathtracer.cpp
    opencode/pt/bvh/bvh.cpp
    opencode/pt/scene.cpp
    opencode/pt/scene/builder.cpp
  PUBLIC
    opencode/pathtracer/version.h
    opencode/anari/device.h
    opencode/anari/library.h
    opencode/pt/bvh/bvh.h
    opencode/pt/camera.h
    opencode/pt/geom/mesh_intersector.h
    opencode/pt/geom/triangle_intersect.h
    opencode/pt/geom/triangle_mesh.h
    opencode/pt/material.h
    opencode/pt/math.h
    opencode/pt/math_aabb.h
    opencode/pt/pathtracer.h
    opencode/pt/ray.h
    opencode/pt/rng.h
    opencode/pt/scene.h
    opencode/pt/scene/builder.h
    opencode/pt/sphere.h
)
