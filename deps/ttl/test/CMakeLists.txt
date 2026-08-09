include(GoogleTest)

file(
    GLOB_RECURSE TTL_PUBLIC_HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../include/ttl/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/../include/ttl/*.cuh"
)

add_custom_target(ttl_public_headers)
foreach(ttl_public_header IN LISTS TTL_PUBLIC_HEADERS)
    file(RELATIVE_PATH ttl_public_header_relative "${CMAKE_CURRENT_LIST_DIR}/../include" "${ttl_public_header}")
    string(REPLACE "/" "_" ttl_public_header_target "${ttl_public_header_relative}")
    string(REPLACE "." "_" ttl_public_header_target "${ttl_public_header_target}")
    set(ttl_public_header_target "header_${ttl_public_header_target}")

    if(ttl_public_header_relative MATCHES "\\.cuh$")
        set(ttl_public_header_extension cu)
    else()
        set(ttl_public_header_extension cpp)
    endif()

    set(
        ttl_public_header_source
        "${CMAKE_CURRENT_BINARY_DIR}/header_check/${ttl_public_header_target}.${ttl_public_header_extension}"
    )
    file(GENERATE OUTPUT "${ttl_public_header_source}" CONTENT "#include <${ttl_public_header_relative}>\n")

    add_library(${ttl_public_header_target} OBJECT EXCLUDE_FROM_ALL "${ttl_public_header_source}")
    add_dependencies(ttl_public_headers ${ttl_public_header_target})
    target_compile_features(${ttl_public_header_target} PRIVATE cxx_std_20 cuda_std_20)
    ttl_enable_warnings(${ttl_public_header_target})
    target_link_libraries(${ttl_public_header_target} PRIVATE ttl::ttl)
endforeach()

add_library(
    ttl_test_support STATIC
    support/test_environment.cpp
    support/test_tensor.cpp
)
target_compile_features(ttl_test_support PUBLIC cxx_std_20 cuda_std_20)
ttl_enable_warnings(ttl_test_support)
target_link_libraries(ttl_test_support PUBLIC ttl::ttl GTest::gtest)
target_include_directories(ttl_test_support PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
set_target_properties(ttl_test_support PROPERTIES CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF)

add_custom_target(ttl_build_tests)
add_dependencies(ttl_build_tests ttl_public_headers)

function(ttl_add_test target_name)
    set(options CUDA MULTI_GPU SERIAL SLOW)
    set(one_value_args TIMEOUT)
    set(multi_value_args SOURCES LABELS)
    cmake_parse_arguments(TTL_TEST "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT TTL_TEST_SOURCES)
        message(FATAL_ERROR "ttl_add_test(${target_name}) requires SOURCES")
    endif()

    add_executable(${target_name} EXCLUDE_FROM_ALL ${TTL_TEST_SOURCES})
    add_dependencies(ttl_build_tests ${target_name})
    target_compile_features(${target_name} PRIVATE cxx_std_20 cuda_std_20)
    ttl_enable_warnings(${target_name})
    target_link_libraries(${target_name} PRIVATE ttl_test_support GTest::gtest_main)
    target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/include")
    set_target_properties(
        ${target_name}
        PROPERTIES
            CXX_EXTENSIONS OFF
            CUDA_EXTENSIONS OFF
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test"
    )

    set(ttl_test_labels ttl ${TTL_TEST_LABELS})
    if(TTL_TEST_CUDA)
        list(APPEND ttl_test_labels cuda)
    else()
        list(APPEND ttl_test_labels host)
    endif()
    if(TTL_TEST_MULTI_GPU)
        list(APPEND ttl_test_labels multi_gpu)
    endif()
    if(TTL_TEST_SLOW)
        list(APPEND ttl_test_labels slow)
    endif()

    if(NOT TTL_TEST_TIMEOUT)
        if(TTL_TEST_SLOW OR TTL_TEST_MULTI_GPU)
            set(TTL_TEST_TIMEOUT 180)
        else()
            set(TTL_TEST_TIMEOUT 60)
        endif()
    endif()

    # GoogleTestAddTests.cmake expands TEST_PROPERTIES once before emitting the
    # CTest script.  Preserve the semicolon list through that expansion so
    # CTest sees one LABELS property containing multiple labels.
    string(REPLACE ";" "\\\\;" ttl_test_labels_escaped "${ttl_test_labels}")
    set(ttl_test_properties LABELS "${ttl_test_labels_escaped}" TIMEOUT "${TTL_TEST_TIMEOUT}")
    if(TTL_TEST_SERIAL OR TTL_TEST_MULTI_GPU)
        list(APPEND ttl_test_properties RUN_SERIAL TRUE)
    endif()

    gtest_discover_tests(
        ${target_name}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        PROPERTIES ${ttl_test_properties}
    )
endfunction()

ttl_add_test(
    ttl_contract_tests
    SOURCES
        contract/public_api_contract_test.cpp
    LABELS contract unit
)

ttl_add_test(
    ttl_host_tests
    SOURCES
        host/common_test.cpp
        host/tensor_value_test.cpp
    LABELS unit
)

ttl_add_test(
    ttl_runtime_tests
    CUDA
    SOURCES
        runtime/runtime_test.cpp
        runtime/kernel_launch_test.cu
    LABELS runtime integration
)

ttl_add_test(
    ttl_tensor_tests
    CUDA
    SOURCES
        tensor/tensor_layout_test.cpp
    LABELS tensor unit
)

ttl_add_test(
    ttl_ops_tests
    CUDA
    SOURCES
        ops/creation_elementwise_test.cpp
        ops/reduction_indexing_test.cpp
        ops/linalg_normalization_test.cpp
        ops/random_selection_test.cpp
    LABELS ops unit
    TIMEOUT 120
)

ttl_add_test(
    ttl_integration_tests
    CUDA
    SOURCES
        integration/async_graph_test.cpp
    LABELS integration
    TIMEOUT 120
)

ttl_add_test(
    ttl_distributed_tests
    CUDA
    MULTI_GPU
    SOURCES
        distributed/distributed_test.cpp
    LABELS distributed integration
)

add_custom_target(
    ttl_check_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_CURRENT_BINARY_DIR}" --output-on-failure -L ttl
    DEPENDS ttl_build_tests
    USES_TERMINAL
)
