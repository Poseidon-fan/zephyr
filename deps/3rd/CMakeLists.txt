# fmt 12.1.0 (407c905e45ad75fc29bf0f9bb7c5c2fd3475976f)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FMT_DOC OFF)
set(FMT_INSTALL OFF)
set(FMT_TEST OFF)
set(FMT_SYSTEM_HEADERS ON)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

# spdlog 1.17.0 (79524ddd08a4ec981b7fea76afd08ee05f83755d)
set(SPDLOG_BUILD_SHARED OFF)
set(SPDLOG_BUILD_PIC ON)
set(SPDLOG_BUILD_EXAMPLE OFF)
set(SPDLOG_BUILD_EXAMPLE_HO OFF)
set(SPDLOG_BUILD_TESTS OFF)
set(SPDLOG_BUILD_TESTS_HO OFF)
set(SPDLOG_BUILD_BENCH OFF)
set(SPDLOG_BUILD_WARNINGS OFF)
set(SPDLOG_INSTALL OFF)
set(SPDLOG_SYSTEM_INCLUDES ON)
set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_USE_STD_FORMAT OFF)
add_subdirectory(spdlog EXCLUDE_FROM_ALL)

# nlohmann/json 3.11.3 is used while parsing checkpoint metadata. Keep the
# dependency in this directory so builds do not depend on a host installation
# or fetch sources during configuration.
set(JSON_BuildTests OFF)
set(JSON_Install OFF)
add_subdirectory(json EXCLUDE_FROM_ALL)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/GoogleTestProvider.cmake")
