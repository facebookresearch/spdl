#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
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
  static SingleDecodingResult async_decode_image(
      const std::string& src,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc);

  static SingleDecodingResult async_decode_image_nvdec(
      const std::string& src,
      const int cuda_device_index,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt);

  ////////////////////////////////////////////////////////////////////////////////
  // Batch image
  ////////////////////////////////////////////////////////////////////////////////
  static MultipleDecodingResult async_batch_decode_image(
      const std::vector<std::string>& srcs,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc);

  static MultipleDecodingResult async_batch_decode_image_nvdec(
      const std::vector<std::string>& srcs,
      const int cuda_device_index,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt);

  ////////////////////////////////////////////////////////////////////////////////
  // Audio / Video
  ////////////////////////////////////////////////////////////////////////////////
  static MultipleDecodingResult async_decode(
      const enum MediaType type,
      const std::string& src,
      const std::vector<std::tuple<double, double>>& timestamps,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const DecodeConfig& decode_cfg,
      const std::string& filter_desc);

  static MultipleDecodingResult async_decode_nvdec(
      const std::string& src,
      const std::vector<std::tuple<double, double>>& timestamps,
      const int cuda_device_index,
      const std::shared_ptr<SourceAdoptor>& adoptor,
      const IOConfig& io_cfg,
      const CropArea& crop,
      int width,
      int height,
      const std::optional<std::string>& pix_fmt);
};

////////////////////////////////////////////////////////////////////////////////
// Future for single decoding result
////////////////////////////////////////////////////////////////////////////////
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

  std::unique_ptr<DecodedFrames> get();

  friend decoding;
};

////////////////////////////////////////////////////////////////////////////////
// Future for multiple decoding result
////////////////////////////////////////////////////////////////////////////////
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

  std::vector<std::unique_ptr<DecodedFrames>> get(bool strict = true);

  friend decoding;
};

} // namespace spdl::core
