message(STATUS "########################################")
message(STATUS "# Configuring CUDA")
message(STATUS "########################################")

enable_language(CUDA)

set(cuda_components cuda_driver cudart)

if (SPDL_USE_NVJPEG)
  list(APPEND cuda_components nvjpeg)
endif()

message(STATUS "Components: ${cuda_components}")
find_package(CUDAToolkit REQUIRED COMPONENT ${cuda_components})
