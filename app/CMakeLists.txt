add_executable(opencode_pathtracer_app main.cpp)

target_link_libraries(opencode_pathtracer_app PRIVATE anari_library_opencode_pathtracer)

target_include_directories(opencode_pathtracer_app PRIVATE "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include")

add_executable(opencode_anari_smoke anari_smoke.cpp)

target_link_libraries(opencode_anari_smoke PRIVATE anari_library_opencode_pathtracer anari)

target_include_directories(opencode_anari_smoke PRIVATE "${CMAKE_BINARY_DIR}/_deps/anari-src/src/anari/include")

add_executable(opencode_pt_render pt_render.cpp)

target_link_libraries(opencode_pt_render PRIVATE anari_library_opencode_pathtracer)

add_executable(opencode_pt_debug_ray pt_debug_ray.cpp)

target_link_libraries(opencode_pt_debug_ray PRIVATE anari_library_opencode_pathtracer)
