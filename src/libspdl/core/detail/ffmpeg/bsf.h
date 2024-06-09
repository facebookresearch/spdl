#pragma once

#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/generator.h"

namespace spdl::core::detail {

class BitStreamFilter {
  AVBSFContextPtr bsf_ctx;

 public:
  BitStreamFilter(const std::string& name, AVCodecParameters* codec_par);

  Generator<AVPacketPtr> filter(AVPacket* packet);
  AVCodecParameters* get_output_codec_par();
};

} // namespace spdl::core::detail
