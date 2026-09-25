add_library(
    zephyr_executor
    OBJECT
    causal_lm.cpp
    embedding.cpp
    executor.cpp
    worker.cpp
)

target_include_directories(
    zephyr_executor
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_executor PUBLIC cxx_std_20)
target_link_libraries(
    zephyr_executor
    PRIVATE
        fmt::fmt
        ttl::ttl
        Threads::Threads
        zephyr::project_warnings
)

set_target_properties(
    zephyr_executor
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_executor>
    PARENT_SCOPE
)
