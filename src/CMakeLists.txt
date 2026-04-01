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

# --------------------------------------------------------------------------
# Photon source files — split into CUDA-compiled and CXX-only groups.
#
# When Kokkos CUDA is enabled (COMPILE_AS_CMAKE_LANGUAGE mode), photon's
# pathtracer core files are compiled as CUDA via nvcc + MSVC host compiler.
# ANARI/helium files MUST remain CXX because CUDA's float4 conflicts with
# linalg::float4 used by the ANARI SDK.
# --------------------------------------------------------------------------

# All photon source files compiled as CUDA when the CUDA backend is enabled.
# nvcc uses MSVC as host compiler (-ccbin cl.exe) so C++ code compiles fine.
set(PHOTON_CUDA_SOURCES
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
  photon/anari/PhotonDevice.cpp
  photon/anari/SceneFromAnari.cpp
  photon/anari/device.cpp
  photon/anari/library.cpp
  photon/anari/entrypoint.cpp
  photon/pt/version.cpp
  photon/pbrt/pbrt_parser.cpp
  photon/pbrt/pbrt_to_photon.cpp
  photon/pbrt/ply_reader.cpp
  photon/pbrt/image_loader.cpp
)

# DeviceImpl.cpp is ANARI SDK source — keep as CXX to avoid any
# potential conflicts with CUDA headers in upstream ANARI code.
set(PHOTON_CXX_SOURCES
  ${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/DeviceImpl.cpp
)

if(OPENCODE_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
  # Compile all photon sources as CUDA (nvcc with MSVC host compiler)
  set_source_files_properties(${PHOTON_CUDA_SOURCES}
    PROPERTIES LANGUAGE CUDA
  )
  # Extended lambda support required for KOKKOS_LAMBDA in device code
  set_source_files_properties(${PHOTON_CUDA_SOURCES}
    PROPERTIES COMPILE_OPTIONS "--expt-extended-lambda;--expt-relaxed-constexpr"
  )
  # DeviceImpl.cpp stays CXX — undefine KOKKOS_ENABLE_CUDA so Kokkos
  # headers don't pull in CUDA backend code that requires nvcc.
  set_source_files_properties(${PHOTON_CXX_SOURCES}
    PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "/UKOKKOS_ENABLE_CUDA"
  )
endif()

target_sources(anari_library_photon
  PRIVATE
    ${PHOTON_CUDA_SOURCES}
    ${PHOTON_CXX_SOURCES}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include/photon/pt/version.h
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
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/pbrt_scene.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/pbrt_parser.h
    ${PROJECT_SOURCE_DIR}/include/photon/pbrt/pbrt_to_photon.h
     ${PROJECT_SOURCE_DIR}/include/photon/pbrt/ply_reader.h
     ${PROJECT_SOURCE_DIR}/include/photon/pbrt/image_loader.h
 )

target_include_directories(anari_library_photon PRIVATE
  ${anari_SOURCE_DIR}/external/stb_image
)

if(PHOTON_EMBREE_READY)
  target_include_directories(anari_library_photon PUBLIC ${EMBREE_INCLUDE_DIRS})
  target_link_libraries(anari_library_photon PUBLIC ${EMBREE_LIBRARIES})
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_EMBREE)
endif()

if(PHOTON_OPTIX_READY)
  target_include_directories(anari_library_photon PUBLIC ${PHOTON_OPTIX_INCLUDE_DIR})
  # Only link cuda_driver — Kokkos already links cudart (static).
  # Linking both CUDA::cudart (dynamic) and cudart_static causes LNK2005.
  target_link_libraries(anari_library_photon PUBLIC CUDA::cuda_driver)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OPTIX)

  set(OPTIX_PTX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/photon/pt/backend/optix/photon_optix.cu")
  set(OPTIX_PTX_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/photon_optix.ptx")
  set(OPTIX_PTX_EMBED  "${CMAKE_CURRENT_BINARY_DIR}/photon_optix_ptx.cpp")

  add_custom_command(
    OUTPUT ${OPTIX_PTX_OUTPUT}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
      -ptx
      -I${PHOTON_OPTIX_INCLUDE_DIR}
      -I${CUDAToolkit_INCLUDE_DIRS}
      --use_fast_math
      -O3
      ${OPTIX_PTX_SOURCE}
      -o ${OPTIX_PTX_OUTPUT}
    DEPENDS ${OPTIX_PTX_SOURCE}
    COMMENT "Compiling OptiX PTX kernel"
  )

  set(OPTIX_EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/embed_ptx.py")
  file(WRITE ${OPTIX_EMBED_SCRIPT}
"import sys
ptx = open(sys.argv[1], 'r').read()
with open(sys.argv[2], 'w') as f:
    f.write('namespace photon::pt {\\n')
    f.write('extern const char photon_optix_ptx[] = R\"PTX(' + ptx + ')PTX\";\\n')
    f.write('extern const unsigned long long photon_optix_ptx_size = sizeof(photon_optix_ptx) - 1;\\n')
    f.write('}\\n')
")

  add_custom_command(
    OUTPUT ${OPTIX_PTX_EMBED}
    COMMAND ${Python3_EXECUTABLE} ${OPTIX_EMBED_SCRIPT} ${OPTIX_PTX_OUTPUT} ${OPTIX_PTX_EMBED}
    DEPENDS ${OPTIX_PTX_OUTPUT}
    COMMENT "Embedding OptiX PTX into C++ source"
  )

  target_sources(anari_library_photon PRIVATE ${OPTIX_PTX_EMBED})
  # Embedded PTX is pure C++ (string literal), must not be compiled as CUDA
  if(OPENCODE_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
    set_source_files_properties(${OPTIX_PTX_EMBED} PROPERTIES LANGUAGE CXX)
  endif()
endif()

if(PHOTON_HIPRT_READY)
  target_link_libraries(anari_library_photon PUBLIC hiprt::hiprt)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_HIPRT)
endif()

if(PHOTON_OIDN_READY)
  target_compile_definitions(anari_library_photon PUBLIC PHOTON_HAS_OIDN)
  if(OpenImageDenoise_FOUND)
    target_link_libraries(anari_library_photon PUBLIC OpenImageDenoise)
  else()
    target_include_directories(anari_library_photon PUBLIC ${PHOTON_OIDN_INCLUDE})
    target_link_libraries(anari_library_photon PUBLIC ${PHOTON_OIDN_LIB})
  endif()
endif()

