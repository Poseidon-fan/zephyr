add_library(zephyr_scheduler OBJECT scheduler.cpp length_bucket_scheduler.cpp paged_scheduler.cpp)

target_include_directories(
    zephyr_scheduler
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_scheduler PUBLIC cxx_std_20)
target_link_libraries(zephyr_scheduler PRIVATE zephyr::project_warnings)

set_target_properties(
    zephyr_scheduler
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_scheduler>
    PARENT_SCOPE
)
