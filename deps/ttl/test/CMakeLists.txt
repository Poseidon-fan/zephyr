include(GoogleTest)

file(GLOB_RECURSE TTL_PUBLIC_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/../include/ttl/*.hpp")

file(
    GLOB_RECURSE TTL_TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cu"
)

add_custom_target(ttl_build_tests)
add_custom_target(ttl_public_headers)
add_dependencies(ttl_build_tests ttl_public_headers)

foreach(ttl_public_header IN LISTS TTL_PUBLIC_HEADERS)
    file(RELATIVE_PATH ttl_public_header_relative "${CMAKE_CURRENT_LIST_DIR}/../include" "${ttl_public_header}")
    string(REPLACE "/" "_" ttl_public_header_target "${ttl_public_header_relative}")
    string(REPLACE ".hpp" "" ttl_public_header_target "${ttl_public_header_target}")
    set(ttl_public_header_target "header_${ttl_public_header_target}")
    set(ttl_public_header_source "${CMAKE_CURRENT_BINARY_DIR}/header_check/${ttl_public_header_target}.cpp")
    file(GENERATE OUTPUT "${ttl_public_header_source}" CONTENT "#include <${ttl_public_header_relative}>\n")

    add_library(${ttl_public_header_target} OBJECT EXCLUDE_FROM_ALL "${ttl_public_header_source}")
    add_dependencies(ttl_public_headers ${ttl_public_header_target})
    target_compile_features(${ttl_public_header_target} PRIVATE cxx_std_20)
    ttl_enable_warnings(${ttl_public_header_target})
    target_link_libraries(${ttl_public_header_target} PRIVATE ttl::ttl)
endforeach()

add_custom_target(
    ttl_check_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_CURRENT_BINARY_DIR}" --output-on-failure -L ttl
    DEPENDS ttl_build_tests
    USES_TERMINAL
)

foreach(ttl_test_source IN LISTS TTL_TEST_SOURCES)
    file(RELATIVE_PATH ttl_test_relative_path "${CMAKE_CURRENT_SOURCE_DIR}" "${ttl_test_source}")
    get_filename_component(ttl_test_category "${ttl_test_relative_path}" DIRECTORY)
    string(REPLACE "/" "_" ttl_test_name "${ttl_test_relative_path}")
    string(REGEX REPLACE "\\.(cpp|cu)$" "" ttl_test_name "${ttl_test_name}")
    string(REPLACE "/" "_" ttl_test_category "${ttl_test_category}")
    if(ttl_test_category STREQUAL "")
        set(ttl_test_category general)
    endif()

    add_executable(${ttl_test_name} EXCLUDE_FROM_ALL "${ttl_test_source}")
    add_dependencies(ttl_build_tests ${ttl_test_name})

    target_compile_features(${ttl_test_name} PRIVATE cxx_std_20 cuda_std_20)
    ttl_enable_warnings(${ttl_test_name})
    target_link_libraries(${ttl_test_name} PRIVATE ttl::ttl GTest::gmock_main)
    target_include_directories(
        ${ttl_test_name}
        PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${CMAKE_CURRENT_LIST_DIR}/../src/include"
    )

    set_target_properties(
        ${ttl_test_name}
        PROPERTIES
            CXX_EXTENSIONS OFF
            CUDA_EXTENSIONS OFF
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test"
    )

    gtest_discover_tests(
        ${ttl_test_name}
        EXTRA_ARGS --gtest_catch_exceptions=0
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            LABELS "ttl_${ttl_test_category}"
            TIMEOUT 30
    )
endforeach()
