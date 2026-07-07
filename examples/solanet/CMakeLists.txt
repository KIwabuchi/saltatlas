function(setup_solanet_example_id_type target_name id_type_name)
    target_compile_definitions(
        ${target_name}
        PRIVATE "SALTATLAS_SOLANET_EXAMPLE_ID_TYPE=${id_type_name}"
    )
endfunction()

function(add_solanet_fe_id_typed_exe target_name fe_type_name id_type_name)
    set(typed_target_name "${target_name}_${id_type_name}_id_${fe_type_name}_features")
    add_general_exe(${typed_target_name} "${target_name}.cpp")
    target_compile_definitions(
        ${typed_target_name}
        PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${fe_type_name}"
    )
    setup_solanet_example_id_type(${typed_target_name} ${id_type_name})
    set(target_name ${typed_target_name} PARENT_SCOPE)
endfunction()

function(add_solanet_fe_id_typed_ygm_exe target_name fe_type_name id_type_name)
    set(typed_target_name "${target_name}_${id_type_name}_id_${fe_type_name}_features")
    add_general_exe(${typed_target_name} "${target_name}.cpp")
    setup_mpi_target(${typed_target_name})
    target_link_libraries(
        ${typed_target_name} PRIVATE ${SALTATLAS_CEREAL_TARGET} ygm::ygm ${BOOST_LIBS})
    target_compile_definitions(
        ${typed_target_name}
        PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${fe_type_name}"
    )
    setup_solanet_example_id_type(${typed_target_name} ${id_type_name})
    set(target_name ${typed_target_name} PARENT_SCOPE)
endfunction()

if (SALTATLAS_USE_APU)
    foreach(id_type_name IN ITEMS uint32_t uint64_t)
        add_solanet_fe_id_typed_exe(run_solanet_single_apu_nndescent float ${id_type_name})
        setup_hip_target(${target_name})

        add_solanet_fe_id_typed_exe(run_solanet_single_apu_nndescent uint8_t ${id_type_name})
        setup_hip_target(${target_name})

        add_solanet_fe_id_typed_exe(run_hipvs_single_apu_nndescent float ${id_type_name})
        setup_hip_target(${target_name})
        setup_hipvs_target(${target_name})

        add_solanet_fe_id_typed_exe(run_hipvs_single_apu_nndescent uint8_t ${id_type_name})
        setup_hip_target(${target_name})
        setup_hipvs_target(${target_name})

        add_solanet_fe_id_typed_ygm_exe(run_solanet_apu float ${id_type_name})
        setup_hip_target(${target_name})
        setup_hipvs_target(${target_name})

        add_solanet_fe_id_typed_ygm_exe(run_solanet_apu uint8_t ${id_type_name})
        setup_hip_target(${target_name})
        setup_hipvs_target(${target_name})
    endforeach()
endif()
