file(
    GLOB_RECURSE ZEPHYR_CORE_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

add_library(zephyr_core STATIC ${ZEPHYR_CORE_SOURCES})
add_library(zephyr::core ALIAS zephyr_core)

target_include_directories(
    zephyr_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(zephyr_core PUBLIC cxx_std_20)
target_link_libraries(zephyr_core PRIVATE ttl::ttl zephyr::project_warnings)

set_target_properties(
    zephyr_core
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
