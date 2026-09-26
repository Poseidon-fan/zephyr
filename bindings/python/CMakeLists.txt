find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.0 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # scikit-build-core injects pybind11_DIR from its build env; fall back to
    # asking the active interpreter for pip-installed pybind11.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(pybind11 3.0 CONFIG REQUIRED)
endif()

file(
    GLOB_RECURSE ZEPHYR_PYTHON_BINDING_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

pybind11_add_module(_C MODULE ${ZEPHYR_PYTHON_BINDING_SOURCES})
add_dependencies(_C clangd-db)
target_compile_features(_C PRIVATE cxx_std_20)
target_link_libraries(_C PRIVATE zephyr::core zephyr::project_warnings)

set_target_properties(
    _C
    PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

install(
    TARGETS _C
    LIBRARY DESTINATION zephyr
    RUNTIME DESTINATION zephyr
)
