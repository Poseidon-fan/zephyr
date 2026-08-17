add_library(
    zephyr_common
    OBJECT
    exception.cpp
    macros.cpp
)

target_include_directories(
    zephyr_common
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_common PUBLIC cxx_std_20)
target_link_libraries(zephyr_common PRIVATE zephyr::project_warnings)

set_target_properties(
    zephyr_common
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_common>
    PARENT_SCOPE
)
