#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVDEC
#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#include "libspdl/core/detail/nvdec/utils.h"
#endif

namespace spdl::core {
namespace {
#ifdef SPDL_USE_NVDEC
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
    std::function<void(NvDecFramesWrapperPtr<media_type>)> set_result,
    std::function<void(std::string)> notify_exception,
    PacketsWrapperPtr<media_type> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr executor) {
#ifndef SPDL_USE_NVDEC
  auto task = folly::coro::co_invoke(
      []() -> folly::coro::Task<NvDecFramesWrapperPtr<media_type>> {
        SPDL_FAIL("SPDL is not compiled with NVDEC support.");
      });
#else
  ThreadPoolExecutorPtr e;
  auto exe = detail::get_demux_executor_high_prio(e);
  auto task = folly::coro::co_invoke(
      [=](PacketsPtr<media_type>&& pkts)
          -> folly::coro::Task<NvDecFramesWrapperPtr<media_type>> {
        validate_nvdec_params(cuda_device_index, crop, width, height);
        init_cuda();
        if constexpr (media_type == MediaType::Video) {
          pkts = co_await detail::apply_bsf(std::move(pkts)).scheduleOn(exe);
        }
        auto frames = co_await detail::decode_nvdec<media_type>(
            std::move(pkts), cuda_device_index, crop, width, height, pix_fmt);
        co_return wrap<media_type, NvDecFramesPtr>(std::move(frames));
      },
      packets->unwrap());
#endif

  return detail::execute_task_with_callback<NvDecFramesWrapperPtr<media_type>>(
      std::move(task),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
}

template FuturePtr async_decode_nvdec(
    std::function<void(NvDecFramesWrapperPtr<MediaType::Video>)> set_result,
    std::function<void(std::string)> notify_exception,
    PacketsWrapperPtr<MediaType::Video> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr async_decode_nvdec(
    std::function<void(NvDecFramesWrapperPtr<MediaType::Image>)> set_result,
    std::function<void(std::string)> notify_exception,
    PacketsWrapperPtr<MediaType::Image> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor);
} // namespace spdl::core
