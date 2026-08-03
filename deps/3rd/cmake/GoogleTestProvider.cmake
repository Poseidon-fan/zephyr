include_guard(GLOBAL)

function(zephyr_require_googletest)
    if(TARGET GTest::gtest OR TARGET GTest::gmock_main)
        if(TARGET GTest::gtest AND TARGET GTest::gmock_main)
            return()
        endif()
        message(FATAL_ERROR "GoogleTest was only partially provided; expected GTest::gtest and GTest::gmock_main")
    endif()

    include(FetchContent)

    set(INSTALL_GTEST OFF)
    set(BUILD_GMOCK ON)
    set(gtest_force_shared_crt ON)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.tar.gz
        URL_HASH SHA256=745c55415660044610f7fcd3af7a6420d5de16a7dbb9ebfe2e131275676232be
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)

    if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gmock_main)
        message(FATAL_ERROR "The configured GoogleTest source did not provide the required imported targets")
    endif()
endfunction()
