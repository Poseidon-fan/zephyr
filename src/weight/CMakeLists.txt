add_library(
    zephyr_weight
    OBJECT
    checkpoint.cpp
    weight_builder.cpp
)

target_include_directories(
    zephyr_weight
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_weight PUBLIC cxx_std_20)
target_link_libraries(
    zephyr_weight
    PRIVATE
        nlohmann_json::nlohmann_json
        ttl::ttl
        zephyr::project_warnings
)

set_target_properties(
    zephyr_weight
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_weight>
    PARENT_SCOPE
)
