add_executable(photon_version_test version_test.cpp)
add_executable(photon_pt_smoke_test pt_smoke_test.cpp)
add_executable(photon_math_test math_test.cpp)
add_executable(photon_sampling_test sampling_test.cpp)
add_executable(photon_mesh_test mesh_test.cpp)
add_executable(photon_disney_bsdf_test disney_bsdf_test.cpp)

target_link_libraries(photon_version_test PRIVATE anari_library_photon)

target_link_libraries(photon_pt_smoke_test PRIVATE anari_library_photon)

target_link_libraries(photon_math_test PRIVATE anari_library_photon)
target_link_libraries(photon_sampling_test PRIVATE anari_library_photon)
target_link_libraries(photon_mesh_test PRIVATE anari_library_photon)
target_link_libraries(photon_disney_bsdf_test PRIVATE anari_library_photon)

add_test(NAME photon_version_test COMMAND $<TARGET_FILE:photon_version_test>)
add_test(NAME photon_pt_smoke_test COMMAND $<TARGET_FILE:photon_pt_smoke_test>)
add_test(NAME photon_math_test COMMAND $<TARGET_FILE:photon_math_test>)
add_test(NAME photon_sampling_test COMMAND $<TARGET_FILE:photon_sampling_test>)
add_test(NAME photon_mesh_test COMMAND $<TARGET_FILE:photon_mesh_test>)
add_test(NAME photon_disney_bsdf_test COMMAND $<TARGET_FILE:photon_disney_bsdf_test>)
