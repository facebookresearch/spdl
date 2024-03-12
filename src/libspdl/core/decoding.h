#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/result.h>
#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {

using DecodeImageResult = Result<FramesPtr, MediaType::Image>;
using BatchDecodeAudioResult = Results<FramesPtr, MediaType::Audio>;
using BatchDecodeVideoResult = Results<FramesPtr, MediaType::Video>;
using BatchDecodeImageResult = Results<FramesPtr, MediaType::Image>;

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
  static DecodeImageResult decode_image(
      const std::string& src,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      std::shared_ptr<ThreadPoolExecutor> demux_executor,
      std::shared_ptr<ThreadPoolExecutor> decode_executor);

  ///
  /// Decode one single image asynchronously using NVDEC.
  ///
  static DecodeImageResult decode_image_nvdec(
      const std::string& src,
      const int cuda_device_index,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt,
      std::shared_ptr<ThreadPoolExecutor> demux_executor,
      std::shared_ptr<ThreadPoolExecutor> decode_executor);

  ////////////////////////////////////////////////////////////////////////////////
  // Batch image
  ////////////////////////////////////////////////////////////////////////////////

  ///
  /// Decode multiple images asynchronously using FFmpeg.
  ///
  static BatchDecodeImageResult batch_decode_image(
      const std::vector<std::string>& srcs,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      std::shared_ptr<ThreadPoolExecutor> demux_executor,
      std::shared_ptr<ThreadPoolExecutor> decode_executor);

  ///
  /// Decode multiple images asynchronously using NVDEC.
  ///
  static BatchDecodeImageResult batch_decode_image_nvdec(
      const std::vector<std::string>& srcs,
      const int cuda_device_index,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt,
      std::shared_ptr<ThreadPoolExecutor> demux_executor,
      std::shared_ptr<ThreadPoolExecutor> decode_executor);

  ////////////////////////////////////////////////////////////////////////////////
  // Audio / Video
  ////////////////////////////////////////////////////////////////////////////////

  ///
  /// Decode multiple clips of the given video/audio asynchronously using
  /// FFmpeg.
  ///
  template <MediaType media_type>
  static Results<FramesPtr, media_type> decode(
      const std::string& src,
      const std::vector<std::tuple<double, double>>& timestamps,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      std::shared_ptr<ThreadPoolExecutor> demux_executor,
      std::shared_ptr<ThreadPoolExecutor> decode_executor);

  ///
  /// Decode multiple clips of the given video asynchronously using NVDEC.
  ///
  static BatchDecodeVideoResult decode_nvdec(
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
      std::shared_ptr<ThreadPoolExecutor> decode_executor);
};

} // namespace spdl::core
