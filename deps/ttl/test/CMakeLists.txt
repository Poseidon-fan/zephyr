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

    set(ttl_layer_labels contract unit integration system)
    set(ttl_layer_count 0)
    foreach(ttl_layer IN LISTS ttl_layer_labels)
        if(ttl_layer IN_LIST TTL_TEST_LABELS)
            math(EXPR ttl_layer_count "${ttl_layer_count} + 1")
        endif()
    endforeach()
    if(NOT ttl_layer_count EQUAL 1)
        message(FATAL_ERROR "${target_name} must specify exactly one layer label: contract, unit, integration, or system")
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
    if(TTL_TEST_CUDA)
        # CUDA tests share an explicitly named logical resource.  The resource name is independent of the number,
        # model, or topology of GPUs on the host and prevents concurrent tests from corrupting one another's runtime.
        list(APPEND ttl_test_properties RESOURCE_LOCK ttl_cuda)
    endif()
    if(TTL_TEST_SERIAL OR TTL_TEST_MULTI_GPU)
        list(APPEND ttl_test_properties RUN_SERIAL TRUE)
    endif()

    gtest_discover_tests(
        ${target_name}
        EXTRA_ARGS --gtest_catch_exceptions=0
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
        PROPERTIES ${ttl_test_properties}
    )
endfunction()

ttl_add_test(
    ttl_contract_tests
    SOURCES
        contract/public_api_contract_test.cpp
    LABELS contract
)

ttl_add_test(
    ttl_common_tests
    SOURCES
        unit/common/common_test.cpp
        unit/common/error_test.cpp
    LABELS unit common
)

ttl_add_test(
    ttl_tensor_host_tests
    SOURCES
        unit/tensor/dtype_test.cpp
        unit/tensor/shape_scalar_test.cpp
        unit/tensor/value_test.cpp
    LABELS unit tensor
)

ttl_add_test(
    ttl_runtime_host_tests
    SOURCES
        unit/runtime/cuda_check_test.cpp
        unit/runtime/memory_primitives_test.cpp
        unit/runtime/runtime_boundary_test.cpp
    LABELS unit runtime
)

ttl_add_test(
    ttl_distributed_unit_tests
    SOURCES
        unit/distributed/collective_common_test.cpp
    LABELS unit distributed
)

ttl_add_test(
    ttl_tensor_tests
    CUDA
    SOURCES
        unit/tensor/layout_test.cpp
        unit/tensor/tensor_layout_test.cpp
    LABELS unit tensor
)

ttl_add_test(
    ttl_creation_tests
    CUDA
    SOURCES
        unit/ops/creation_test.cpp
    LABELS unit ops creation
    TIMEOUT 120
)

ttl_add_test(
    ttl_elementwise_tests
    CUDA
    SOURCES
        unit/ops/elementwise_test.cpp
    LABELS unit ops elementwise
    TIMEOUT 120
)

ttl_add_test(
    ttl_linalg_tests
    CUDA
    SOURCES
        unit/ops/linalg_test.cpp
    LABELS unit ops linalg
    TIMEOUT 120
)

ttl_add_test(
    ttl_attention_random_tests
    CUDA
    SOURCES
        unit/ops/attention_random_test.cpp
    LABELS unit ops attention random
    TIMEOUT 120
)

ttl_add_test(
    ttl_reduction_indexing_tests
    CUDA
    SOURCES
        unit/ops/reduction_test.cpp
    LABELS unit ops indexing reduction scan
    TIMEOUT 120
)

ttl_add_test(
    ttl_operator_contract_tests
    CUDA
    SOURCES
        unit/ops/operator_contract_smoke_test.cpp
    LABELS unit ops smoke
    TIMEOUT 120
)

ttl_add_test(
    ttl_selection_tests
    CUDA
    SOURCES
        unit/ops/random_selection_test.cpp
    LABELS unit ops selection random
    TIMEOUT 120
)

ttl_add_test(
    ttl_operator_edge_tests
    CUDA
    SOURCES
        unit/ops/operator_edge_case_test.cpp
    LABELS unit ops edge
    TIMEOUT 120
)

ttl_add_test(
    ttl_dtype_matrix_tests
    CUDA
    SOURCES
        unit/ops/dtype_matrix_test.cpp
    LABELS unit ops dtype
    TIMEOUT 120
)

ttl_add_test(
    ttl_output_contract_tests
    CUDA
    SOURCES
        unit/ops/output_contract_test.cpp
    LABELS unit ops output
    TIMEOUT 120
)

ttl_add_test(
    ttl_runtime_tests
    CUDA
    SOURCES
        unit/runtime/kernel_launch_test.cu
        unit/runtime/public_kernel_launch_test.cu
    LABELS unit runtime
)

ttl_add_test(
    ttl_runtime_integration_tests
    CUDA
    SOURCES
        integration/runtime/runtime_api_test.cpp
        integration/runtime/runtime_tensor_test.cpp
        integration/runtime/graph_test.cpp
        integration/runtime/memory_copy_test.cpp
    LABELS integration runtime
    TIMEOUT 120
)

ttl_add_test(
    ttl_runtime_system_tests
    CUDA
    SOURCES
        system/runtime/async_graph_test.cpp
    LABELS system runtime
    TIMEOUT 120
)

ttl_add_test(
    ttl_distributed_integration_tests
    CUDA
    MULTI_GPU
    SOURCES
        integration/distributed/collective_test.cpp
        integration/distributed/nccl_rank_local_test.cpp
    LABELS integration distributed
    TIMEOUT 180
)

ttl_add_test(
    ttl_distributed_system_tests
    CUDA
    MULTI_GPU
    SOURCES
        system/distributed/distributed_workflow_test.cpp
    LABELS system distributed
    TIMEOUT 180
)

add_custom_target(
    ttl_check_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_CURRENT_BINARY_DIR}" --output-on-failure -L ttl
    DEPENDS ttl_build_tests
    USES_TERMINAL
)
