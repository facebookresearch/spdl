#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

namespace spdl::core {

class DecodingResultFuture;

#define ASYNC_DECODE                                             \
  DecodingResultFuture async_decode(                             \
      const enum MediaType type,                                 \
      const std::string& src,                                    \
      const std::vector<std::tuple<double, double>>& timestamps, \
      const std::shared_ptr<SourceAdoptor>& adoptor,             \
      const IOConfig& io_cfg,                                    \
      const DecodeConfig& decode_cfg,                            \
      const std::string& filter_desc);

ASYNC_DECODE;

class DecodingResultFuture {
  using ResultType = std::vector<std::unique_ptr<FrameContainer>>;

  struct Impl;

  Impl* pimpl = nullptr;

  DecodingResultFuture(Impl* impl);

 public:
  DecodingResultFuture() = delete;
  DecodingResultFuture(const DecodingResultFuture&) = delete;
  DecodingResultFuture& operator=(const DecodingResultFuture&) = delete;
  DecodingResultFuture(DecodingResultFuture&&) noexcept;
  DecodingResultFuture& operator=(DecodingResultFuture&&) noexcept;
  ~DecodingResultFuture();

  ResultType get();

  friend ASYNC_DECODE;
};

#undef ASYNC_DECODE

} // namespace spdl::core
