add_library(cuvs_utils STATIC cuvs_utils.cu)
setup_cuvs_target(cuvs_utils)
target_include_directories(cuvs_utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
setup_common_compile_options(cuvs_utils)
