/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/adaptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include <fmt/core.h>
#include <glog/logging.h>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
}

namespace spdl::core {
namespace detail {

namespace {
class Bytes {
  std::string_view buffer;
  uint64_t pos = 0;

 public:
  explicit Bytes(std::string_view data) : buffer(std::move(data)) {}

  Bytes(const Bytes&) = delete;
  Bytes& operator=(const Bytes&) = delete;
  Bytes(Bytes&& other) noexcept = delete;
  Bytes& operator=(Bytes&& other) noexcept = delete;

 private:
  int read_packet(uint8_t* buf, int buf_size) {
    if (int64_t remaining = buffer.size() - pos; remaining < buf_size) {
      buf_size = static_cast<int>(remaining);
    }
    if (buf_size <= 0) {
      return AVERROR_EOF;
    }
    memcpy(buf, buffer.data() + pos, buf_size);
    pos += buf_size;
    return buf_size;
  }

  int64_t seek(int64_t offset, int whence) {
    auto size = buffer.size();
    switch (whence) {
      case AVSEEK_SIZE:
        return static_cast<int64_t>(size);
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos += offset;
        break;
      case SEEK_END:
        pos = buffer.size() + offset;
        break;
      default:
        LOG(ERROR) << "Unexpected whence value was found: " << whence;
        return -1;
    }
    if (pos > size) {
      pos = size;
    }
    return pos;
  }

 public:
  static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    return static_cast<Bytes*>(opaque)->read_packet(buf, buf_size);
  }

  static int64_t seek(void* opaque, int64_t offset, int whence) {
    return static_cast<Bytes*>(opaque)->seek(offset, whence);
  }
  const char* get_src() const {
    return buffer.data();
  }
};

class BytesInterface : public DataInterface {
  Bytes obj;
  AVIOContextPtr io_ctx;
  AVFormatInputContextPtr fmt_ctx;

 public:
  BytesInterface(std::string_view data, const DemuxConfig& dmx_cfg)
      : obj(data),
        io_ctx(get_io_ctx(
            &obj,
            dmx_cfg.buffer_size,
            Bytes::read_packet,
            Bytes::seek)),
        fmt_ctx(get_input_format_ctx(
            io_ctx.get(),
            dmx_cfg.format,
            dmx_cfg.format_options)) {
    auto src = fmt::format("<Bytes: {}>", (void*)obj.get_src());
    fmt_ctx->url = av_strdup(src.c_str());
  }

  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};
} // namespace
} // namespace detail

DataInterfacePtr BytesAdaptor::get_interface(
    std::string_view data,
    const DemuxConfig& dmx_cfg) const {
  return DataInterfacePtr(new detail::BytesInterface{data, dmx_cfg});
}

} // namespace spdl::core
