add_library(
    zephyr_sampler
    OBJECT
    sampler.cpp
    sampler.cu
)

target_include_directories(
    zephyr_sampler
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_sampler PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(zephyr_sampler PRIVATE ttl::ttl zephyr::project_warnings)

set_target_properties(
    zephyr_sampler
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_sampler>
    PARENT_SCOPE
)
