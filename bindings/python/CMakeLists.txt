find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

file(
    GLOB_RECURSE ZEPHYR_PYTHON_BINDING_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

pybind11_add_module(_C MODULE ${ZEPHYR_PYTHON_BINDING_SOURCES})
target_compile_features(_C PRIVATE cxx_std_20)
target_link_libraries(_C PRIVATE zephyr::core zephyr::project_warnings)

set_target_properties(
    _C
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

install(
    TARGETS _C
    LIBRARY DESTINATION zephyr
    RUNTIME DESTINATION zephyr
)
