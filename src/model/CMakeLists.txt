add_library(
    zephyr_model
    OBJECT
    causal_lm/model.cpp
    causal_lm/qwen3.cpp
    causal_lm/qwen3_config.cpp
)

target_include_directories(
    zephyr_model
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_model PUBLIC cxx_std_20)
target_link_libraries(
    zephyr_model
    PRIVATE
        nlohmann_json::nlohmann_json
        ttl::ttl
        zephyr::project_warnings
)

set_target_properties(
    zephyr_model
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_model>
    PARENT_SCOPE
)
