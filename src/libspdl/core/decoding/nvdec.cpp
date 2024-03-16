#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/decoding.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/result.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVDEC
#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/nvdec/decoding.h"
#include "libspdl/core/detail/nvdec/utils.h"
#endif

#include <folly/experimental/coro/Task.h>
#include <folly/futures/Future.h>
#include <folly/logging/xlog.h>

namespace spdl::core {

using folly::SemiFuture;
using folly::coro::Task;

namespace {
#ifdef SPDL_USE_NVDEC
Task<NvDecImageFramesPtr> image_decode_task_nvdec(
    const std::string src,
    const int cuda_device_index,
    const SourceAdoptorPtr adoptor,
    const IOConfig io_cfg,
    const CropArea crop,
    int width,
    int height,
    const std::optional<std::string> pix_fmt,
    ThreadPoolExecutorPtr decode_executor) {
  auto exec = detail::get_decode_executor(decode_executor);
  auto packet =
      co_await detail::demux_image(src, std::move(adoptor), std::move(io_cfg));
  auto task = detail::decode_nvdec<MediaType::Image>(
      std::move(packet), cuda_device_index, crop, width, height, pix_fmt);
  SemiFuture<NvDecImageFramesPtr> future =
      std::move(task).scheduleOn(exec).start();
  co_return co_await std::move(future);
}

Task<std::vector<SemiFuture<NvDecImageFramesPtr>>>
batch_image_decode_task_nvdec(
    const std::vector<std::string> srcs,
    const int cuda_device_index,
    const SourceAdoptorPtr adoptor,
    const IOConfig io_cfg,
    const CropArea crop,
    int width,
    int height,
    const std::optional<std::string> pix_fmt,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
  std::vector<SemiFuture<NvDecImageFramesPtr>> futures;
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

Task<std::vector<SemiFuture<NvDecVideoFramesPtr>>> stream_decode_task_nvdec(
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    const int cuda_device_index,
    const SourceAdoptorPtr adoptor,
    const IOConfig io_cfg,
    const CropArea crop,
    int width,
    int height,
    const std::optional<std::string> pix_fmt,
    ThreadPoolExecutorPtr decode_executor) {
  std::vector<SemiFuture<NvDecVideoFramesPtr>> futures;
  {
    auto exec = detail::get_decode_executor(decode_executor);
    auto demuxer = detail::stream_demux<MediaType::Video>(
        src, timestamps, std::move(adoptor), std::move(io_cfg));
    while (auto result = co_await demuxer.next()) {
      auto filtered = co_await detail::apply_bsf(*result);
      auto task = detail::decode_nvdec<MediaType::Video>(
          std::move(filtered), cuda_device_index, crop, width, height, pix_fmt);
      futures.emplace_back(std::move(task).scheduleOn(exec).start());
    }
  }
  co_return std::move(futures);
}
#endif
} // namespace

DecodeNvDecResult<MediaType::Image> decoding::decode_image_nvdec(
    const std::string& src,
    const int cuda_device_index,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  detail::validate_nvdec_params(cuda_device_index, crop, width, height);
  detail::init_cuda();
  return DecodeNvDecResult<MediaType::Image>{
      new DecodeNvDecResult<MediaType::Image>::Impl{
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

BatchDecodeNvDecResult<MediaType::Video> decoding::decode_video_nvdec(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const int cuda_device_index,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  if (timestamps.size() == 0) {
    SPDL_FAIL("At least one timestamp must be provided.");
  }

  detail::validate_nvdec_params(cuda_device_index, crop, width, height);
  detail::init_cuda();

  return BatchDecodeNvDecResult<MediaType::Video>{
      new BatchDecodeNvDecResult<MediaType::Video>::Impl{
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

BatchDecodeNvDecResult<MediaType::Image> decoding::batch_decode_image_nvdec(
    const std::vector<std::string>& srcs,
    const int cuda_device_index,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    const CropArea& crop,
    int width,
    int height,
    const std::optional<std::string>& pix_fmt,
    ThreadPoolExecutorPtr demux_executor,
    ThreadPoolExecutorPtr decode_executor) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  if (srcs.size() == 0) {
    SPDL_FAIL("At least one source must be provided.");
  }

  detail::validate_nvdec_params(cuda_device_index, crop, width, height);
  detail::init_cuda();

  return BatchDecodeNvDecResult<MediaType::Image>{
      new BatchDecodeNvDecResult<MediaType::Image>::Impl{
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
