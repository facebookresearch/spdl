#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"

#ifdef SPDL_USE_NVDEC
#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#include "libspdl/core/detail/nvdec/utils.h"
#endif

namespace spdl::core {

template <MediaType media_type>
FuturePtr decode_nvdec_async(
    std::function<void(std::optional<NvDecFramesPtr<media_type>>)> set_result,
    std::function<void()> notify_exception,
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  detail::validate_nvdec_params(cuda_device_index, crop, width, height);
  detail::init_cuda();

  return detail::execute_task_with_callback<NvDecFramesPtr<media_type>>(
      detail::decode_nvdec<media_type>(
          std::move(packets), cuda_device_index, crop, width, height, pix_fmt),
      std::move(set_result),
      std::move(notify_exception),
      detail::get_decode_executor(executor));
#endif
}

template FuturePtr decode_nvdec_async(
    std::function<void(std::optional<NvDecVideoFramesPtr>)> set_result,
    std::function<void()> notify_exception,
    VideoPacketsPtr packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor);

template FuturePtr decode_nvdec_async(
    std::function<void(std::optional<NvDecImageFramesPtr>)> set_result,
    std::function<void()> notify_exception,
    ImagePacketsPtr packets,
    int cuda_device_index,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor);

} // namespace spdl::core
