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
