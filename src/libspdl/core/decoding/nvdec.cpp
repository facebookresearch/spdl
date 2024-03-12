#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/result.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVDEC
#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#endif

#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::Task;

namespace {
#ifdef SPDL_USE_NVDEC
Task<FramesPtr> image_decode_task_nvdec(
    const std::string src,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const CropArea crop,
    int width,
    int height,
    const std::optional<std::string> pix_fmt,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  auto exec = detail::get_decode_executor(decode_executor);
  auto packet =
      co_await detail::demux_image(src, std::move(adoptor), std::move(io_cfg));
  auto task = detail::decode_packets_nvdec(
      std::move(packet),
      cuda_device_index,
      crop,
      width,
      height,
      pix_fmt,
      /*is_image*/ true);
  SemiFuture<FramesPtr> future = std::move(task).scheduleOn(exec).start();
  co_return co_await std::move(future);
}

Task<std::vector<SemiFuture<FramesPtr>>> batch_image_decode_task_nvdec(
    const std::vector<std::string> srcs,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const CropArea crop,
    int width,
    int height,
    const std::optional<std::string> pix_fmt,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  std::vector<SemiFuture<FramesPtr>> futures;
  for (auto& src : srcs) {
    futures.emplace_back(
        image_decode_task_nvdec(
            src,
            cuda_device_index,
            adoptor,
            io_cfg,
            crop,
            width,
            height,
            pix_fmt,
            decode_executor)
            .scheduleOn(detail::get_demux_executor(demux_executor))
            .start());
  }
  co_return std::move(futures);
}

Task<std::vector<SemiFuture<FramesPtr>>> stream_decode_task_nvdec(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg,
    const CropArea crop,
    int width,
    int height,
    const std::optional<std::string> pix_fmt,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
  std::vector<SemiFuture<FramesPtr>> futures;
  {
    auto exec = detail::get_decode_executor(decode_executor);
    auto demuxer = detail::stream_demux_nvdec(
        src, timestamps, std::move(adoptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto task = detail::decode_packets_nvdec(
          *std::move(result), cuda_device_index, crop, width, height, pix_fmt);
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }
  co_return std::move(futures);
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
#endif
} // namespace

DecodeImageResult decoding::decode_image_nvdec(
    const std::string& src,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  validate_nvdec_params(cuda_device_index, crop, width, height);
  init_cuda();
  return DecodeImageResult{new DecodeImageResult::Impl{
      image_decode_task_nvdec(
          src,
          cuda_device_index,
          adoptor,
          io_cfg,
          crop,
          width,
          height,
          pix_fmt,
          decode_executor)
          .scheduleOn(detail::get_demux_executor(demux_executor))
          .start()}};
#endif
}

BatchDecodeVideoResult decoding::decode_nvdec(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  if (timestamps.size() == 0) {
    SPDL_FAIL("At least one timestamp must be provided.");
  }

  validate_nvdec_params(cuda_device_index, crop, width, height);
  init_cuda();

  return BatchDecodeVideoResult{new BatchDecodeVideoResult::Impl{
      {src},
      timestamps,
      stream_decode_task_nvdec(
          src,
          timestamps,
          cuda_device_index,
          adoptor,
          io_cfg,
          crop,
          width,
          height,
          pix_fmt,
          decode_executor)
          .scheduleOn(detail::get_demux_executor(demux_executor))
          .start()}};
#endif
}

BatchDecodeImageResult decoding::batch_decode_image_nvdec(
    const std::vector<std::string>& srcs,
    const int cuda_device_index,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    std::shared_ptr<ThreadPoolExecutor> demux_executor,
    std::shared_ptr<ThreadPoolExecutor> decode_executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  if (srcs.size() == 0) {
    SPDL_FAIL("At least one source must be provided.");
  }

  validate_nvdec_params(cuda_device_index, crop, width, height);
  init_cuda();

  return BatchDecodeImageResult{new BatchDecodeImageResult::Impl{
      srcs,
      {},
      batch_image_decode_task_nvdec(
          srcs,
          cuda_device_index,
          adoptor,
          io_cfg,
          crop,
          width,
          height,
          pix_fmt,
          demux_executor,
          decode_executor)
          .scheduleOn(detail::get_demux_executor(demux_executor))
          .start()}};

#endif
}
} // namespace spdl::core
