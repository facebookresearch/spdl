#include <libspdl/coro/decoding.h>

#include "libspdl/coro/detail/executor.h"
#include "libspdl/coro/detail/future.h"

#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVCODEC
#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#include "libspdl/core/detail/nvdec/utils.h"
#endif

namespace spdl::coro {
template <MediaType media_type>
FuturePtr async_decode_nvdec(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor) {
#ifndef SPDL_USE_NVCODEC
  auto task = folly::coro::co_invoke([]() -> folly::coro::Task<CUDABufferPtr> {
    SPDL_FAIL("SPDL is not compiled with NVDEC support.");
  });
#else
  ThreadPoolExecutorPtr e;
  auto exe = detail::get_demux_executor_high_prio(e);
  auto task = folly::coro::co_invoke(
      [=](PacketsPtr<media_type> pkts) -> folly::coro::Task<CUDABufferPtr> {
        co_return spdl::core::decode_packets_nvdec<media_type>(
            std::move(pkts),
            cuda_device_index,
            crop,
            width,
            height,
            pix_fmt,
            cuda_stream,
            cuda_allocator);
      },
      std::move(packets));
#endif

  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

template FuturePtr async_decode_nvdec(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<MediaType::Video> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr async_decode_nvdec(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    PacketsPtr<MediaType::Image> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr demux_executor);

FuturePtr async_batch_decode_image_nvdec(
    std::function<void(CUDABufferPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::vector<PacketsPtr<MediaType::Image>>&& packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    bool strict,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator,
    ThreadPoolExecutorPtr executor) {
#ifndef SPDL_USE_NVCODEC
  auto task = folly::coro::co_invoke([]() -> folly::coro::Task<CUDABufferPtr> {
    SPDL_FAIL("SPDL is not compiled with NVDEC support.");
  });
#else
  auto task = folly::coro::co_invoke(
      [=](std::vector<PacketsPtr<MediaType::Image>>&& pkts)
          -> folly::coro::Task<CUDABufferPtr> {
        co_return spdl::core::decode_packets_nvdec(
            std::move(pkts),
            cuda_device_index,
            crop,
            width,
            height,
            pix_fmt,
            strict,
            cuda_stream,
            cuda_allocator);
      },
      std::move(packets));
#endif

  return detail::execute_task_with_callback(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}
} // namespace spdl::coro
