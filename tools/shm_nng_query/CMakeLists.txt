function(add_omp_exe example_name type_name)
    set(example_exe "${example_name}_${type_name}_features")
    add_executable(${example_exe} "${example_name}.cpp")
    target_link_libraries(${example_exe} PUBLIC saltatlas)
    target_link_libraries(${example_exe} PRIVATE Threads::Threads)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${example_exe} PRIVATE OpenMP::OpenMP_CXX)
    endif ()
    target_compile_definitions(${example_exe}
            PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${type_name}")
endfunction()

# TODO: remove Metall dependency
if (SALTATLAS_USE_METALL)
    add_omp_exe(run_query float)
    add_omp_exe(run_query double)
    add_omp_exe(run_query uint8_t)
endif ()