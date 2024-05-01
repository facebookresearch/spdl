#include <libspdl/core/conversion.h>

#ifdef SPDL_USE_CUDA
#include "libspdl/core/detail/cuda.h"
#endif
#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/future.h"
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
    const std::optional<cuda_allocator_fn>& cuda_allocator,
    const std::optional<cuda_deleter_fn>& cuda_deleter,
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

  std::unique_ptr<CUDABuffer> ret;

  if (cuda_allocator) {
    if (!cuda_deleter) {
      SPDL_FAIL(
          "When allocator is provided, deleter and stream must be provided as well.");
    }
    ret = cuda_buffer(
        buffer->shape,
        cuda_stream,
        cuda_device_index,
        buffer->elem_class,
        buffer->depth,
        cuda_allocator.value(),
        cuda_deleter.value());
  } else {
    ret = cuda_buffer(
        buffer->shape,
        static_cast<CUstream>(reinterpret_cast<void*>(cuda_stream)),
        cuda_device_index,
        buffer->elem_class,
        buffer->depth);
  }

  size_t size = buffer->depth * prod(buffer->shape);
  CHECK_CUDA(
      cudaMemcpy(ret->data(), buffer->data(), size, cudaMemcpyHostToDevice),
      "Failed to copy data from host to device.");

  return ret;
#endif
}

} // namespace spdl::core
