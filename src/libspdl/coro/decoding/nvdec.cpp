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
namespace {
#ifdef SPDL_USE_NVCODEC
void init_cuda() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    TRACE_EVENT("nvdec", "cudaGetDeviceCount");
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count), "Failed to fetch the device count.");
    if (count == 0) {
      SPDL_FAIL("No CUDA device was found.");
    }
  });
}

void validate_nvdec_params(
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height) {
  if (cuda_device_index < 0) {
    SPDL_FAIL(fmt::format(
        "cuda_device_index must be non-negative. Found: {}",
        cuda_device_index));
  }
  if (crop.left < 0) {
    SPDL_FAIL(
        fmt::format("crop.left must be non-negative. Found: {}", crop.left));
  }
  if (crop.top < 0) {
    SPDL_FAIL(
        fmt::format("crop.top must be non-negative. Found: {}", crop.top));
  }
  if (crop.right < 0) {
    SPDL_FAIL(
        fmt::format("crop.right must be non-negative. Found: {}", crop.right));
  }
  if (crop.bottom < 0) {
    SPDL_FAIL(fmt::format(
        "crop.bottom must be non-negative. Found: {}", crop.bottom));
  }
  if (width > 0 && width % 2) {
    SPDL_FAIL(fmt::format("width must be positive and even. Found: {}", width));
  }
  if (height > 0 && height % 2) {
    SPDL_FAIL(
        fmt::format("height must be positive and even. Found: {}", height));
  }
}
#endif
} // namespace

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
        validate_nvdec_params(cuda_device_index, crop, width, height);
        init_cuda();

        if constexpr (media_type == MediaType::Video) {
          pkts = co_await folly::coro::co_invoke(
                     [=](PacketsPtr<media_type> pkts)
                         -> folly::coro::Task<PacketsPtr<media_type>> {
                       co_return apply_bsf(std::move(pkts));
                     },
                     std::move(pkts))
                     .scheduleOn(exe);
        }
        co_return spdl::core::detail::decode_nvdec<media_type>(
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
        validate_nvdec_params(cuda_device_index, crop, width, height);
        init_cuda();
        co_return spdl::core::detail::decode_nvdec(
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
