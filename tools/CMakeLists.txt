# Standalone rendering regression test.
#
# We compile the needed photon sources directly into this executable rather
# than linking the SHARED anari_library_photon DLL.  This avoids Windows
# DLL symbol-export issues under CUDA and mirrors how the project's own
# app/ targets work (they are also skipped under CUDA because CXX cannot
# include Kokkos headers propagated from a CUDA-compiled shared library).

set(PHOTON_TEST_SOURCES
  ${PROJECT_SOURCE_DIR}/src/photon/pt/pathtracer.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/bvh/bvh.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/scene.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/scene/builder.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/environment_map.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/denoiser.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/backend/kokkos_backend.cpp
  ${PROJECT_SOURCE_DIR}/src/photon/pt/backend/backend_factory.cpp
)

add_executable(photon_render_test render_test.cpp ${PHOTON_TEST_SOURCES})

target_include_directories(photon_render_test PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(photon_render_test PRIVATE Kokkos::kokkos)

# When Kokkos CUDA is enabled, compile all sources as CUDA
# (same pattern as src/CMakeLists.txt for photon sources)
if(OPENCODE_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
  set_source_files_properties(render_test.cpp ${PHOTON_TEST_SOURCES}
    PROPERTIES
      LANGUAGE CUDA
      COMPILE_OPTIONS "--expt-extended-lambda;--expt-relaxed-constexpr"
  )
endif()
