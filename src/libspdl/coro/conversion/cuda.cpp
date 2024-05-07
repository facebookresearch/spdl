#include <libspdl/coro/conversion.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

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

BufferPtr convert_to_cuda(
    BufferPtr buffer,
    int cuda_device_index,
    uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& allocator,
    bool async) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
  if (buffer->is_cuda()) {
    return buffer;
  }
  if (async) {
    SPDL_FAIL("Async conversion is not supported.");
  }

  TRACE_EVENT("decoding", "core::convert_to_cuda");

  auto ret = cuda_buffer(
      buffer->shape,
      cuda_device_index,
      cuda_stream,
      allocator,
      buffer->elem_class,
      buffer->depth);

  size_t size = buffer->depth * prod(buffer->shape);
  CHECK_CUDA(
      cudaMemcpy(ret->data(), buffer->data(), size, cudaMemcpyHostToDevice),
      "Failed to copy data from host to device.");

  return ret;
#endif
}

} // namespace spdl::core
