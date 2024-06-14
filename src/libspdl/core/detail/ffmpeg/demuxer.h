#pragma once

#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/generator.h"

namespace spdl::core::detail {

class Demuxer {
  AVFormatContext* fmt_ctx;

 public:
  explicit Demuxer(AVFormatContext* fmt_ctx);

  Generator<AVPacketPtr> demux();
};

} // namespace spdl::core::detail
