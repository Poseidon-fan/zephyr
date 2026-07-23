option(TTL_BUILD_TESTING "Build the TTL test suite" OFF)

file(
    GLOB_RECURSE TTL_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(ttl STATIC ${TTL_SOURCES})
add_library(ttl::ttl ALIAS ttl)

target_include_directories(
    ttl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(ttl PUBLIC cxx_std_20)
target_link_libraries(
    ttl
    PRIVATE
        zephyr::project_warnings
        CUDA::cublas
        CUDA::cublasLt
        CUDA::curand
        NCCL::NCCL
)

set_target_properties(
    ttl
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

if(TTL_BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
