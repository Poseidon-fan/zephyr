cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to compile")
endif()

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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# scikit-build-core puts the build dir under build/{wheel_tag}, which clangd does
# not search; mirror the compilation database where clangd expects to find it.
add_custom_target(
    clangd-db ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${PROJECT_SOURCE_DIR}/build/compile_commands.json"
    COMMENT "Exporting compile_commands.json for clangd"
    VERBATIM
)

option(ZEPHYR_BUILD_PYTHON "Build the Zephyr Python extension" ON)
option(ZEPHYR_BUILD_TESTING "Build the Zephyr test suite" OFF)

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
if(ZEPHYR_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()
if(ZEPHYR_BUILD_TESTING)
    enable_testing()
    zephyr_require_googletest()
    add_subdirectory(test)
endif()
