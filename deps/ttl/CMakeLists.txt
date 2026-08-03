cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to compile")
endif()

project(ttl LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TTL_BUILD_TESTING "Build the TTL test suite" OFF)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TTLCompilerOptions)
include(TTLDependencies)
ttl_validate_cuda_architectures()

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
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_compile_features(ttl PUBLIC cxx_std_20 cuda_std_20)
ttl_enable_warnings(ttl)
target_link_libraries(
    ttl
    PUBLIC
        CUDA::cudart
        NCCL::NCCL
    PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        CUDA::curand
)

set_target_properties(
    ttl
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

if(TTL_BUILD_TESTING)
    enable_testing()

    if(NOT COMMAND zephyr_require_googletest)
        include("${CMAKE_CURRENT_SOURCE_DIR}/../3rd/cmake/GoogleTestProvider.cmake")
    endif()
    zephyr_require_googletest()

    add_subdirectory(test)
endif()
