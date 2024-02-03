#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace spdl::detail {

void cuda_check_impl(
    const cudaError_t err,
    const char* filename,
    const char* function_name,
    const int line_number);

} // namespace spdl::detail

#define CUDA_CHECK(EXPR)                                             \
  do {                                                               \
    const cudaError_t __err = EXPR;                                  \
    spdl::detail::cuda_check_impl(                                   \
        __err, __FILE__, __func__, static_cast<uint32_t>(__LINE__)); \
  } while (0)
