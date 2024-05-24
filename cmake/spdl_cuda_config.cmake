message(STATUS "########################################")
message(STATUS "# Looking for CUDA (${cuda_components})")
message(STATUS "########################################")

set(cuda_components cuda_driver cudart)

if (SPDL_USE_NVJPEG)
  list(APPEND cuda_components nvjpeg)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED COMPONENT ${cuda_components})
