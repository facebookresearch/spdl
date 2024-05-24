message(STATUS "########################################")
message(STATUS "# Configuring CUDA")
message(STATUS "########################################")

enable_language(CUDA)

set(cuda_components cuda_driver cudart)

if (SPDL_USE_NVJPEG)
  if (SPDL_LINK_STATIC_NVJPEG)
    list(APPEND cuda_components nvjpeg_static)
  else()
    list(APPEND cuda_components nvjpeg)
  endif()
endif()

message(STATUS "Components: ${cuda_components}")
find_package(CUDAToolkit REQUIRED COMPONENT ${cuda_components})

foreach(component ${cuda_components})
  set(target CUDA::${component})
  if(NOT TARGET CUDA::${component})
    message(FATAL_ERROR "There is no target named ${target}")
  else()
    message(STATUS "Found ${target}")
    print_target_properties(${target})
  endif()
endforeach()
