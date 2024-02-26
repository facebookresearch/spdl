#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Image
////////////////////////////////////////////////////////////////////////////////
class SingleDecodingResult;

#define ASYNC_DECODE_IMAGE                           \
  SingleDecodingResult async_decode_image(           \
      const std::string& src,                        \
      const std::shared_ptr<SourceAdoptor>& adoptor, \
      const IOConfig& io_cfg,                        \
      const DecodeConfig& decode_cfg,                \
      const std::string& filter_desc);

ASYNC_DECODE_IMAGE;

#define ASYNC_DECODE_IMAGE_NVDEC                     \
  SingleDecodingResult async_decode_image_nvdec(     \
      const std::string& src,                        \
      const int cuda_device_index,                   \
      const std::shared_ptr<SourceAdoptor>& adoptor, \
      const IOConfig& io_cfg,                        \
      int crop_left,                                 \
      int crop_top,                                  \
      int crop_right,                                \
      int crop_bottom,                               \
      int width,                                     \
      int height,                                    \
      const std::optional<std::string>& pix_fmt)

ASYNC_DECODE_IMAGE_NVDEC;

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

  friend ASYNC_DECODE_IMAGE;
  friend ASYNC_DECODE_IMAGE_NVDEC;
};

#undef ASYNC_DECODE_IMAGE
#undef ASYNC_DECODE_IMAGE_NVDEC

////////////////////////////////////////////////////////////////////////////////
// Audio / Video
////////////////////////////////////////////////////////////////////////////////
class MultipleDecodingResult;

#define ASYNC_BATCH_DECODE_IMAGE                     \
  MultipleDecodingResult async_batch_decode_image(   \
      const std::vector<std::string>& srcs,          \
      const std::shared_ptr<SourceAdoptor>& adoptor, \
      const IOConfig& io_cfg,                        \
      const DecodeConfig& decode_cfg,                \
      const std::string& filter_desc);

ASYNC_BATCH_DECODE_IMAGE;

#define ASYNC_DECODE                                             \
  MultipleDecodingResult async_decode(                           \
      const enum MediaType type,                                 \
      const std::string& src,                                    \
      const std::vector<std::tuple<double, double>>& timestamps, \
      const std::shared_ptr<SourceAdoptor>& adoptor,             \
      const IOConfig& io_cfg,                                    \
      const DecodeConfig& decode_cfg,                            \
      const std::string& filter_desc);

ASYNC_DECODE;

#define ASYNC_DECODE_NVDEC                                       \
  MultipleDecodingResult async_decode_nvdec(                     \
      const std::string& src,                                    \
      const std::vector<std::tuple<double, double>>& timestamps, \
      const int cuda_device_index,                               \
      const std::shared_ptr<SourceAdoptor>& adoptor,             \
      const IOConfig& io_cfg,                                    \
      int crop_left,                                             \
      int crop_top,                                              \
      int crop_right,                                            \
      int crop_bottom,                                           \
      int width,                                                 \
      int height,                                                \
      const std::optional<std::string>& pix_fmt)

ASYNC_DECODE_NVDEC;

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

  friend ASYNC_DECODE;
  friend ASYNC_DECODE_NVDEC;
  friend ASYNC_BATCH_DECODE_IMAGE;
};

#undef ASYNC_DECODE
#undef ASYNC_DECODE_NVDEC
#undef ASYNC_BATCH_DECODE_IMAGE

} // namespace spdl::core
