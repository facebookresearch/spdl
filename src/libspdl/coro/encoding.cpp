#include <libspdl/coro/encoding.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/cuda.h>
#include <libspdl/core/encoding.h>

namespace spdl::coro {

FuturePtr async_encode_image(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    const std::string& pix_fmt,
    const std::optional<EncodeConfig>& encode_cfg,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke([=]() -> folly::coro::Task<int> {
    spdl::core::encode_image(
        uri, data, shape, pix_fmt, encode_cfg.value_or(EncodeConfig{}));
    co_return 0; // dummy
  });
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_encode_executor(executor));
}

FuturePtr async_encode_image_cuda(
    std::function<void(int)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    const std::string& pix_fmt,
    const std::optional<EncodeConfig>& encode_cfg,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke([=]() -> folly::coro::Task<int> {
    auto storage = spdl::core::cp_to_cpu(data, shape);
    spdl::core::encode_image(uri, storage.data(), shape, pix_fmt, encode_cfg);
    co_return 0; // dummy
  });
  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_encode_executor(executor));
}

} // namespace spdl::coro
