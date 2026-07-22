cmake_minimum_required(VERSION 3.24)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" ZEPHYR_VERSION LIMIT_COUNT 1)
string(STRIP "${ZEPHYR_VERSION}" ZEPHYR_VERSION)

if(NOT ZEPHYR_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION must contain a semantic version such as 0.1.0")
endif()

project(
    zephyr
    VERSION "${ZEPHYR_VERSION}"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(NCCL REQUIRED)

add_library(zephyr_project_warnings INTERFACE)
add_library(zephyr::project_warnings ALIAS zephyr_project_warnings)

target_compile_options(
    zephyr_project_warnings
    INTERFACE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wall,-Wextra>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->"
)

add_subdirectory(deps)
add_subdirectory(src)
add_subdirectory(bindings/python)
