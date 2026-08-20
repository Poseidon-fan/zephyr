add_library(
    zephyr_config
    OBJECT
        config.cpp
)

target_include_directories(
    zephyr_config
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_config PUBLIC cxx_std_20)
target_link_libraries(
    zephyr_config
    PUBLIC ttl::ttl
    PRIVATE zephyr::project_warnings
)

set_target_properties(
    zephyr_config
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_config>
    PARENT_SCOPE
)
