include(GoogleTest)

file(GLOB_RECURSE ZEPHYR_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*/*test.cpp")

add_custom_target(build-tests COMMAND ${CMAKE_CTEST_COMMAND} --show-only)
add_custom_target(check-tests COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

foreach(zephyr_test_source ${ZEPHYR_TEST_SOURCES})
    get_filename_component(zephyr_test_filename ${zephyr_test_source} NAME)
    string(REPLACE ".cpp" "" zephyr_test_name ${zephyr_test_filename})

    add_executable(${zephyr_test_name} EXCLUDE_FROM_ALL ${zephyr_test_source})
    add_dependencies(build-tests ${zephyr_test_name})
    add_dependencies(check-tests ${zephyr_test_name})

    gtest_discover_tests(
        ${zephyr_test_name}
        EXTRA_ARGS --gtest_catch_exceptions=0
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        PROPERTIES TIMEOUT 60
    )

    target_compile_features(${zephyr_test_name} PRIVATE cxx_std_20)
    target_link_libraries(
        ${zephyr_test_name}
        PRIVATE
            zephyr::core
            zephyr::project_warnings
            GTest::gtest_main
    )

    set_target_properties(
        ${zephyr_test_name}
        PROPERTIES
            CXX_EXTENSIONS OFF
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test"
    )
endforeach()
