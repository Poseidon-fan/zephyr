add_library(
    zephyr_kv_cache
    OBJECT
    block_hash.cpp
    block_pool.cpp
    cache_engine.cpp
    manager.cpp
)

# xxHash 0.8.1 is compiled privately from its vendored upstream header.
target_include_directories(zephyr_kv_cache SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/deps/3rd/xxhash")

target_include_directories(
    zephyr_kv_cache
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
)

target_compile_features(zephyr_kv_cache PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(
    zephyr_kv_cache
    PRIVATE
        ttl::ttl
        zephyr::project_warnings
)

set_target_properties(
    zephyr_kv_cache
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

set(
    ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:zephyr_kv_cache>
    PARENT_SCOPE
)
