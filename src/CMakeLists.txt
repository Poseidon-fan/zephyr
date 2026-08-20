add_subdirectory(common)
add_subdirectory(config)
add_subdirectory(ir)

add_library(zephyr_core STATIC vector_adder.cpp ${ALL_OBJECT_FILES})
add_library(zephyr::core ALIAS zephyr_core)

target_include_directories(
    zephyr_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(zephyr_core PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(
    zephyr_core
    PUBLIC
        fmt::fmt
        ttl::ttl
    PRIVATE spdlog::spdlog zephyr::project_warnings
)

set_target_properties(
    zephyr_core
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
