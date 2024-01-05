#include <libspdl/detail/cuda.h>

#include <fmt/core.h>

namespace spdl::detail {

void cuda_check_impl(
    const cudaError_t err,
    const char* filename,
    const char* function_name,
    const int line_number) {
  if (cudaSuccess != err) [[unlikely]] {
    throw std::runtime_error(fmt::format(
        "CUDA error: {} ({}:{} - {})",
        cudaGetErrorString(err),
        filename,
        line_number,
        function_name));
  }
}

} // namespace spdl::detail
