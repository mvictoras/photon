add_executable(photon_app main.cpp)

target_link_libraries(photon_app PRIVATE anari_library_photon)

target_include_directories(photon_app PRIVATE "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include")

add_executable(photon_anari_smoke anari_smoke.cpp)

target_link_libraries(photon_anari_smoke PRIVATE anari_library_photon anari)

target_include_directories(photon_anari_smoke PRIVATE "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include")

add_executable(photon_render photon_render.cpp)

target_link_libraries(photon_render PRIVATE anari_library_photon)

add_executable(photon_debug_ray photon_debug_ray.cpp)

target_link_libraries(photon_debug_ray PRIVATE anari_library_photon)

add_executable(photon_anari_render photon_anari_render.cpp)

target_link_libraries(photon_anari_render PRIVATE anari_library_photon anari)

target_include_directories(photon_anari_render PRIVATE "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include")

add_executable(photon_pbrt_render photon_pbrt_render.cpp)

target_link_libraries(photon_pbrt_render PRIVATE anari_library_photon)

if(PHOTON_MPI_READY)
  add_executable(photon_mpi_render photon_mpi_render.cpp)
  target_link_libraries(photon_mpi_render PRIVATE anari_library_photon MPI::MPI_CXX)
endif()

add_executable(photon_scene_render photon_scene_render.cpp)

target_link_libraries(photon_scene_render PRIVATE anari_library_photon anari anari_test_scenes helium)

target_include_directories(photon_scene_render PRIVATE
  "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include"
  "${CMAKE_BINARY_DIR}/_deps/anari-src/src"
  "${CMAKE_BINARY_DIR}/_deps/anari-build/src/anari_test_scenes"
)
