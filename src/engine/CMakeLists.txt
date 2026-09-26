add_library(zephyr_engine OBJECT engine.cpp generation.cpp embedding.cpp)

target_include_directories(zephyr_engine PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>)
target_compile_features(zephyr_engine PUBLIC cxx_std_20)
target_link_libraries(zephyr_engine PRIVATE fmt::fmt ttl::ttl Threads::Threads zephyr::project_warnings)

set_target_properties(
    zephyr_engine
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(ALL_OBJECT_FILES ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_engine> PARENT_SCOPE)
