#include <libspdl/core/adoptor/bytes.h>

#include <folly/logging/xlog.h>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
}

namespace spdl::core {
namespace {

class Bytes {
  std::string_view buffer;
  int64_t pos = 0;

 public:
  Bytes(const std::string& data) : buffer(data.c_str(), data.size()) {}

  Bytes(const Bytes&) = delete;
  Bytes& operator=(const Bytes&) = delete;

  Bytes(Bytes&& other) noexcept {
    *this = std::move(other);
  }

  Bytes& operator=(Bytes&& other) noexcept {
    using std::swap;
    swap(buffer, other.buffer);
    swap(pos, other.pos);
    return *this;
  }

  ~Bytes() = default;

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
  BytesInterface(const std::string& data, const IOConfig& io_cfg)
      : obj(data),
        io_ctx(detail::get_io_ctx(
            &obj,
            io_cfg.buffer_size,
            Bytes::read_packet,
            Bytes::seek)),
        fmt_ctx(detail::get_input_format_ctx(
            io_ctx,
            io_cfg.format,
            io_cfg.format_options)) {}

  ~BytesInterface() {
    detail::free_av_io_ctx(io_ctx);
    detail::free_av_fmt_ctx(fmt_ctx);
  };
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx;
  }
};
} // namespace

void* BytesAdoptor::get(const std::string& data, const IOConfig& io_cfg) const {
  return new BytesInterface{data, io_cfg};
}

} // namespace spdl::core
