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
  int64_t pos = 0;

 public:
  explicit Bytes(std::string_view data) : buffer(std::move(data)) {}

  Bytes(const Bytes&) = delete;
  Bytes& operator=(const Bytes&) = delete;
  Bytes(Bytes&& other) noexcept = delete;
  Bytes& operator=(Bytes&& other) noexcept = delete;

 private:
  int read_packet(uint8_t* buf, int buf_size) {
    if (int remining = buffer.size() - pos; remining < buf_size) {
      buf_size = remining;
    }
    if (!buf_size) {
      return AVERROR_EOF;
    }
    memcpy(buf, buffer.data() + pos, buf_size);
    pos += buf_size;
    return buf_size;
  }

  int64_t seek(int64_t offset, int whence) {
    switch (whence) {
      case AVSEEK_SIZE:
        return static_cast<int64_t>(buffer.size());
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
    return pos;
  }

 public:
  static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    return static_cast<Bytes*>(opaque)->read_packet(buf, buf_size);
  }

  static int64_t seek(void* opaque, int64_t offset, int whence) {
    return static_cast<Bytes*>(opaque)->seek(offset, whence);
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
    std::string url = fmt::format("<Bytes: {}>", (void*)data.data());
    fmt_ctx->url = av_strdup(url.data());
  }

  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};
} // namespace
} // namespace detail

std::unique_ptr<DataInterface> BytesAdaptor::get(
    std::string_view data,
    const DemuxConfig& dmx_cfg) const {
  return std::unique_ptr<DataInterface>(
      new detail::BytesInterface{data, dmx_cfg});
}

} // namespace spdl::core
