#include <libspdl/coro/decoding.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/decoding.h>

namespace spdl::coro {

FuturePtr async_decode_image_nvjpeg(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string_view& data,
    int cuda_device_index,
    const std::string& pix_fmt,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr decode_executor) {
  auto task = folly::coro::co_invoke([=]() -> folly::coro::Task<CUDABufferPtr> {
    co_return spdl::core::decode_image_nvjpeg(
        data, cuda_device_index, pix_fmt, cuda_allocator);
  });
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(decode_executor));
}

} // namespace spdl::coro
