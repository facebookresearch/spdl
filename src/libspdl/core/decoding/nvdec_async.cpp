#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/demuxing.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"

#ifdef SPDL_USE_NVDEC
#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#include "libspdl/core/detail/nvdec/utils.h"
#endif

namespace spdl::core {

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
        detail::validate_nvdec_params(cuda_device_index, crop, width, height);
        detail::init_cuda();
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
