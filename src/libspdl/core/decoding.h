#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {
class SingleDecodingResult;
class MultipleDecodingResult;

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
  static SingleDecodingResult async_decode_image(
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
  static SingleDecodingResult async_decode_image_nvdec(
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
  static MultipleDecodingResult async_batch_decode_image(
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
  static MultipleDecodingResult async_batch_decode_image_nvdec(
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
  static MultipleDecodingResult async_decode(
      const enum MediaType type,
      const std::string& src,
      const std::vector<std::tuple<double, double>>& timestamps,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc,
      std::shared_ptr<ThreadPoolExecutor> demux_executor,
      std::shared_ptr<ThreadPoolExecutor> decode_executor);

  ///
  /// Decode multiple clips of the given video/audio asynchronously using NVDEC.
  ///
  static MultipleDecodingResult async_decode_nvdec(
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

////////////////////////////////////////////////////////////////////////////////
// Future for single decoding result
////////////////////////////////////////////////////////////////////////////////

/// Future-like object that holds the result of single asynchronous decoding
/// operation. Used for decoding images.
class SingleDecodingResult {
  struct Impl;

  Impl* pimpl = nullptr;

  SingleDecodingResult(Impl* impl);

 public:
  SingleDecodingResult() = delete;
  SingleDecodingResult(const SingleDecodingResult&) = delete;
  SingleDecodingResult& operator=(const SingleDecodingResult&) = delete;
  SingleDecodingResult(SingleDecodingResult&&) noexcept;
  SingleDecodingResult& operator=(SingleDecodingResult&&) noexcept;
  ~SingleDecodingResult();

  /// Blocks until the decoding is completed and the frame data is ready.
  /// If the decoding operation fails, throws an exception.
  std::unique_ptr<DecodedFrames> get();

  friend decoding;
};

////////////////////////////////////////////////////////////////////////////////
// Future for multiple decoding result
////////////////////////////////////////////////////////////////////////////////

/// Future-like object that holds the results of multiple asynchronous decoding
/// operation. Used for decoding audio and video clips.
class MultipleDecodingResult {
  struct Impl;

  Impl* pimpl = nullptr;

  MultipleDecodingResult(Impl* impl);

 public:
  MultipleDecodingResult() = delete;
  MultipleDecodingResult(const MultipleDecodingResult&) = delete;
  MultipleDecodingResult& operator=(const MultipleDecodingResult&) = delete;
  MultipleDecodingResult(MultipleDecodingResult&&) noexcept;
  MultipleDecodingResult& operator=(MultipleDecodingResult&&) noexcept;
  ~MultipleDecodingResult();

  /// Blocks until all the decoding operations are completed and the frame
  /// data are ready.
  ///
  /// If a decoding operation fails, and ``strict==true``, then throws one of
  /// the exception thrown from the failed operation.
  ///
  /// If ``strict==false``, then exceptions are not propagated. However, if
  /// there is no decoding result to return, (all the decoding operations fail)
  /// it throws an exception.
  std::vector<std::unique_ptr<DecodedFrames>> get(bool strict = true);

  friend decoding;
};

} // namespace spdl::core
