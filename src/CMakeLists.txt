add_library(opencode_pathtracer STATIC)

target_include_directories(opencode_pathtracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(opencode_pathtracer PUBLIC cxx_std_23)

target_sources(opencode_pathtracer
  PRIVATE
    opencode/pathtracer/version.cpp
  PUBLIC
    opencode/pathtracer/version.h
)
