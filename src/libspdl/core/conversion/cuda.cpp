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

BufferPtr convert_to_cuda(BufferPtr buffer, int cuda_device_index) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
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

FuturePtr async_convert_to_cuda(
    std::function<void(BufferWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    BufferWrapperPtr buffer,
    int cuda_device_index,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [](BufferPtr&& b,
         int device_index) -> folly::coro::Task<BufferWrapperPtr> {
#ifndef SPDL_USE_CUDA
        SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
        co_return wrap(convert_to_cuda(std::move(b), device_index));
#endif
      },
      buffer->unwrap(),
      cuda_device_index);
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

} // namespace spdl::core
