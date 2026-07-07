# Copyright 2019 Lawrence Livermore National Security, LLC and other YGM
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

#
# This function adds an mpi test.
#
function(add_saltatlas_test test_name)
    set(test_source "${test_name}.cpp")
    set(test_exe "${test_name}")
    add_executable(${test_exe} ${test_source})
    target_link_libraries(${test_exe} saltatlas)
endfunction()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common/CMakeLists.txt")
    add_subdirectory(common)
endif ()
if (SALTATLAS_USE_METALL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dnnd/CMakeLists.txt")
    add_subdirectory(dnnd)
endif ()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/neo_dnnd/CMakeLists.txt")
    add_subdirectory(neo_dnnd)
endif ()
