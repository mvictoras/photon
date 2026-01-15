cmake_minimum_required(VERSION 3.25)

project(opencode_pathtracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
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
