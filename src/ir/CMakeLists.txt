add_library(
    zephyr_ir
    OBJECT
        model.cpp
        value.cpp
        operation/core.cpp
        operation/tensor.cpp
)

target_include_directories(
    zephyr_ir
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(zephyr_ir PUBLIC cxx_std_20)
target_link_libraries(
    zephyr_ir
    PUBLIC ttl::ttl
    PRIVATE zephyr::project_warnings
)

set_target_properties(
    zephyr_ir
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_ir>
    PARENT_SCOPE
)
