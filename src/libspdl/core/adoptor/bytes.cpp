#include <libspdl/core/adoptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include <folly/logging/xlog.h>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
}

namespace spdl::core {
namespace detail {
void free_av_io_ctx(AVIOContext* p);
void free_av_fmt_ctx(AVFormatContext* p);

namespace {
class Bytes {
  std::string_view buffer;
  int64_t pos = 0;

 public:
  Bytes(std::string_view data) : buffer(std::move(data)) {}

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
        XLOG(ERR) << "Unexpected whence value was found: " << whence;
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
  AVIOContext* io_ctx;
  AVFormatContext* fmt_ctx;

 public:
  BytesInterface(std::string_view data, const IOConfig& io_cfg)
      : obj(data),
        io_ctx(get_io_ctx(
            &obj,
            io_cfg.buffer_size,
            Bytes::read_packet,
            Bytes::seek)),
        fmt_ctx(get_input_format_ctx(
            io_ctx,
            io_cfg.format,
            io_cfg.format_options)) {}

  ~BytesInterface() {
    free_av_io_ctx(io_ctx);
    free_av_fmt_ctx(fmt_ctx);
  };
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx;
  }
};
} // namespace
} // namespace detail

DataInterface* BytesAdoptor::get(std::string_view data, const IOConfig& io_cfg)
    const {
  return new detail::BytesInterface{data, io_cfg};
}

} // namespace spdl::core
