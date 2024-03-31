#include <libspdl/core/adaptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/logging.h"

#include <folly/logging/xlog.h>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
#include <libavutil/file.h>
}

namespace spdl::core {
namespace detail {

namespace {
class MemoryMappedFile {
  uint8_t* buffer_ = nullptr;
  size_t buffer_size_ = 0;

  int64_t pos_ = 0;

 public:
  MemoryMappedFile(const std::string& path) {
    CHECK_AVERROR(
        av_file_map(path.data(), &buffer_, &buffer_size_, 0, NULL),
        "Failed to map file ({}).",
        path);
  };

  MemoryMappedFile(const MemoryMappedFile&) = delete;
  MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
  MemoryMappedFile(MemoryMappedFile&& other) noexcept = delete;
  MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept = delete;

  ~MemoryMappedFile() {
    av_file_unmap(buffer_, buffer_size_);
  };

 private:
  int read_packet(uint8_t* buf, int buf_size) {
    buf_size = FFMIN(buf_size, buffer_size_ - pos_);
    if (!buf_size) {
      return AVERROR_EOF;
    }
    memcpy(buf, buffer_ + pos_, buf_size);
    pos_ += buf_size;
    return buf_size;
  };

  int64_t seek(int64_t offset, int whence) {
    switch (whence) {
      case AVSEEK_SIZE:
        return buffer_size_;
      case SEEK_SET:
        pos_ = offset;
        break;
      case SEEK_CUR:
        pos_ += offset;
        break;
      case SEEK_END:
        pos_ = buffer_size_ + offset;
        break;
      default:
        XLOG(ERR) << "Unexpected whence value was found: " << whence;
        return -1;
    }
    return pos_;
  };

 public:
  static int read_packet(void* opaque, uint8_t* buf, int buf_size) {
    return static_cast<MemoryMappedFile*>(opaque)->read_packet(buf, buf_size);
  };

  static int64_t seek(void* opaque, int64_t offset, int whence) {
    return static_cast<MemoryMappedFile*>(opaque)->seek(offset, whence);
  };
};

class MMapInterface : public DataInterface {
  MemoryMappedFile obj;
  AVIOContextPtr io_ctx;
  AVFormatInputContextPtr fmt_ctx;

 public:
  MMapInterface(std::string path, const IOConfig& io_cfg)
      : obj(path),
        io_ctx(get_io_ctx(
            &obj,
            io_cfg.buffer_size,
            MemoryMappedFile::read_packet,
            MemoryMappedFile::seek)),
        fmt_ctx(get_input_format_ctx(
            io_ctx.get(),
            io_cfg.format,
            io_cfg.format_options)) {
    fmt_ctx->url = av_strdup(path.data());
  }

  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};
} // namespace
} // namespace detail

DataInterface* MMapAdaptor::get(std::string_view url, const IOConfig& io_cfg)
    const {
  return new detail::MMapInterface(std::string{url}, io_cfg);
};

} // namespace spdl::core
