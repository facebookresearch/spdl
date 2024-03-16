#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/future.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/result.h>
#include <libspdl/core/types.h>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {

/// Decode audio, video or image
template <MediaType media_type>
FuturePtr decode_async(
    std::function<void(std::optional<FFmpegFramesPtr<media_type>>)> set_result,
    std::function<void()> notify_exception,
    PacketsPtr<media_type> packets,
    DecodeConfig decode_cfg,
    std::string filter_desc,
    ThreadPoolExecutorPtr decode_executor);

////////////////////////////////////////////////////////////////////////////////
// Synchronous decodings
////////////////////////////////////////////////////////////////////////////////

template <MediaType media_type>
using DecodeResult = Result<FFmpegFramesPtr<media_type>>;

template <MediaType media_type>
using DecodeNvDecResult = Result<NvDecFramesPtr<media_type>>;

template <MediaType media_type>
using BatchDecodeResult = Results<FFmpegFramesPtr<media_type>>;

template <MediaType media_type>
using BatchDecodeNvDecResult = Results<NvDecFramesPtr<media_type>>;

// Putting all the decoding functions into this utility, static-only class
// so that we can make the whole thing friend of result classes without having
// to repeat the signatures.
//
// This is not really used as a class, so we use lower case for the name.
struct decoding {
  decoding() = delete;

  ////////////////////////////////////////////////////////////////////////////////
  // Image
  ////////////////////////////////////////////////////////////////////////////////

  ///
  /// Decode one single image asynchronously using FFmpeg.
  ///
  static DecodeResult<MediaType::Image> decode_image(
      const std::string& src,
      const SourceAdoptorPtr& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      ThreadPoolExecutorPtr demux_executor,
      ThreadPoolExecutorPtr decode_executor);

  ///
  /// Decode one single image asynchronously using NVDEC.
  ///
  static DecodeNvDecResult<MediaType::Image> decode_image_nvdec(
      const std::string& src,
      const int cuda_device_index,
      const SourceAdoptorPtr& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt,
      ThreadPoolExecutorPtr demux_executor,
      ThreadPoolExecutorPtr decode_executor);

  ////////////////////////////////////////////////////////////////////////////////
  // Batch image
  ////////////////////////////////////////////////////////////////////////////////

  ///
  /// Decode multiple images asynchronously using FFmpeg.
  ///
  static BatchDecodeResult<MediaType::Image> batch_decode_image(
      const std::vector<std::string>& srcs,
      const SourceAdoptorPtr& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      ThreadPoolExecutorPtr demux_executor,
      ThreadPoolExecutorPtr decode_executor);

  ///
  /// Decode multiple images asynchronously using NVDEC.
  ///
  static BatchDecodeNvDecResult<MediaType::Image> batch_decode_image_nvdec(
      const std::vector<std::string>& srcs,
      const int cuda_device_index,
      const SourceAdoptorPtr& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt,
      ThreadPoolExecutorPtr demux_executor,
      ThreadPoolExecutorPtr decode_executor);

  ////////////////////////////////////////////////////////////////////////////////
  // Audio / Video
  ////////////////////////////////////////////////////////////////////////////////

  ///
  /// Decode multiple clips of the given video/audio asynchronously using
  /// FFmpeg.
  ///
  template <MediaType media_type>
  static Results<FFmpegFramesPtr<media_type>> decode(
      const std::string& src,
      const std::vector<std::tuple<double, double>>& timestamps,
      const SourceAdoptorPtr& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      ThreadPoolExecutorPtr demux_executor,
      ThreadPoolExecutorPtr decode_executor);

  ///
  /// Decode multiple clips of the given video asynchronously using NVDEC.
  ///
  static BatchDecodeNvDecResult<MediaType::Video> decode_video_nvdec(
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
      ThreadPoolExecutorPtr decode_executor);
};

} // namespace spdl::core
