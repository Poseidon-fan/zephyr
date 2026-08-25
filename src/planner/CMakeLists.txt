add_library(
    zephyr_planner
    OBJECT
        planner.cpp
        lowering.cpp
)

target_include_directories(zephyr_planner PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(
    zephyr_planner
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_planner PUBLIC cxx_std_20)
target_link_libraries(
    zephyr_planner
    PUBLIC ttl::ttl
    PRIVATE zephyr::project_warnings
)

set_target_properties(
    zephyr_planner
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(ALL_OBJECT_FILES ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_planner> PARENT_SCOPE)
