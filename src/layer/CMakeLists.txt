add_library(
    zephyr_layer
    OBJECT
    activation.cpp
    activation.cu
    embedding.cpp
    linear.cpp
    mlp.cpp
    normalization.cpp
    paged_attention.cpp
    rotary_embedding.cpp
    rotary_embedding.cu
)

target_include_directories(
    zephyr_layer
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_layer PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(
    zephyr_layer
    PRIVATE
        ttl::ttl
        zephyr::project_warnings
)

set_target_properties(
    zephyr_layer
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
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_layer>
    PARENT_SCOPE
)
