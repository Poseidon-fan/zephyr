find_path(
    NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${NCCL_ROOT} ENV NCCL_ROOT
    PATH_SUFFIXES include
)

find_library(
    NCCL_LIBRARY
    NAMES nccl
    HINTS ${NCCL_ROOT} ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64
)

if(NCCL_INCLUDE_DIR)
    foreach(component MAJOR MINOR PATCH)
        file(
            STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_${component}_LINE
            REGEX "^#define NCCL_${component} +[0-9]+$"
        )
        string(
            REGEX REPLACE "^#define NCCL_${component} +([0-9]+)$" "\\1"
            NCCL_VERSION_${component} "${NCCL_VERSION_${component}_LINE}"
        )
    endforeach()

    if(NCCL_VERSION_MAJOR AND NCCL_VERSION_MINOR AND NCCL_VERSION_PATCH)
        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(
        NCCL::NCCL
        PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
