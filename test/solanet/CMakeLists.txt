if (SALTATLAS_USE_APU)
    add_general_exe(test_solanet_apu_uint64_id_instantiation
                   test_solanet_apu_uint64_id_instantiation.cpp)
    target_link_libraries(test_solanet_apu_uint64_id_instantiation PRIVATE saltatlas)
    setup_hip_target(test_solanet_apu_uint64_id_instantiation)
endif ()
