add_library(anari_library_photon SHARED)
set_target_properties(anari_library_photon PROPERTIES OUTPUT_NAME "anari_library_photon")

opencode_enable_warnings(anari_library_photon)

target_link_libraries(anari_library_photon PUBLIC Kokkos::kokkos anari helium)

target_include_directories(anari_library_photon PUBLIC
  "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include"
  "${CMAKE_BINARY_DIR}/_deps/anari-src/src"
)

target_include_directories(anari_library_photon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(anari_library_photon PUBLIC cxx_std_20)

# If Kokkos CUDA is enabled, use nvcc_wrapper for this target only
# (leaves ANARI SDK compiled with the host C++ compiler)
if(Kokkos_ENABLE_CUDA)
  kokkos_compilation(TARGET anari_library_photon)
endif()

target_sources(anari_library_photon
  PRIVATE
    photon/pathtracer/version.cpp
    photon/anari/device.cpp
    photon/anari/library.cpp
    photon/anari/entrypoint.cpp
    photon/anari/PhotonDevice.cpp
    photon/anari/SceneFromAnari.cpp
    ${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/DeviceImpl.cpp
    photon/pt/pathtracer.cpp
    photon/pt/bvh/bvh.cpp
    photon/pt/scene.cpp
    photon/pt/scene/builder.cpp
    photon/pt/environment_map.cpp
    photon/pt/denoiser.cpp
    photon/pt/backend/kokkos_backend.cpp
    photon/pt/backend/embree_backend.cpp
    photon/pt/backend/optix_backend.cpp
    photon/pt/backend/hiprt_backend.cpp
    photon/pt/backend/backend_factory.cpp
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include/photon/pathtracer/version.h
    ${PROJECT_SOURCE_DIR}/include/photon/anari/device.h
    ${PROJECT_SOURCE_DIR}/include/photon/anari/library.h
    photon/anari/PhotonDevice.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/bvh/bvh.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/bvh/bvh_validate.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/camera.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/geom/mesh_intersector.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/geom/triangle_intersect.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/geom/triangle_mesh.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/material.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/disney_bsdf.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/tone_mapping.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_aabb.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_vec2.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/math_mat4.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/color.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/pathtracer.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/ray.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/rng.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/sampling.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/volume.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/light.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/environment_map.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/denoiser.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/texture.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/scene.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/scene/builder.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/sphere.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/ray_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/kokkos_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/embree_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/optix_backend.h
    ${PROJECT_SOURCE_DIR}/include/photon/pt/backend/hiprt_backend.h
 )

