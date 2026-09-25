add_library(
    zephyr_attention
    OBJECT
    cache.cpp
    cache.cu
    paged_decode.cpp
    paged_decode.cu
)

target_include_directories(
    zephyr_attention
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_attention PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(
    zephyr_attention
    PRIVATE
        ttl::ttl
        zephyr::project_warnings
)

set_target_properties(
    zephyr_attention
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
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_attention>
    PARENT_SCOPE
)
