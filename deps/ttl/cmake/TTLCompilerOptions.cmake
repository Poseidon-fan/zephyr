include_guard(GLOBAL)

function(ttl_validate_cuda_architectures)
    foreach(cuda_architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(cuda_architecture STREQUAL "native")
            continue()
        endif()

        if(NOT cuda_architecture MATCHES "^([0-9]+)(-real|-virtual)?$")
            message(FATAL_ERROR "TTL requires explicit SM80+ CUDA architectures; got '${cuda_architecture}'")
        endif()

        if(CMAKE_MATCH_1 LESS 80)
            message(FATAL_ERROR "TTL requires SM80 or newer; got '${cuda_architecture}'")
        endif()
    endforeach()
endfunction()

function(ttl_enable_warnings target_name)
    target_compile_options(
        ${target_name}
        PRIVATE
            "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wall,-Wextra>"
            "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->"
    )
endfunction()
