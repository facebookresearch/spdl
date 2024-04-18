#include <libspdl/core/conversion.h>

#ifdef SPDL_USE_CUDA
#include "libspdl/core/detail/cuda.h"
#endif
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

namespace spdl::core {
namespace {
size_t prod(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto& v : shape) {
    ret *= v;
  }
  return ret;
}
} // namespace

BufferPtr convert_to_cuda(BufferPtr buffer, int cuda_device_index) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else

  TRACE_EVENT("decoding", "core::convert_to_cuda");
  auto ret = cuda_buffer(
      buffer->shape,
      0,
      cuda_device_index,
      buffer->channel_last,
      buffer->elem_class,
      buffer->depth);

  if (buffer->is_cuda()) {
    return buffer;
  }

  size_t size = buffer->depth * prod(buffer->shape);

  CHECK_CUDA(
      cudaMemcpy(ret->data(), buffer->data(), size, cudaMemcpyHostToDevice),
      "Failed to copy data from host to device.");

  return ret;

#endif
}

} // namespace spdl::core
