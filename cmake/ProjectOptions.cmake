include_guard(GLOBAL)

function(opencode_enable_warnings target)
  if(MSVC)
    # CXX-compiled files get full warnings.
    # CUDA-compiled files get warnings passed through nvcc's -Xcompiler.
    # /permissive- is omitted for CUDA because it conflicts with CUDA headers.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
    )
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()
