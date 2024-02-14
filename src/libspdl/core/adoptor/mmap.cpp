#include <libspdl/core/adoptor/mmap.h>

#include <libspdl/core/detail/ffmpeg/logging.h>

#include <folly/logging/xlog.h>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
#include <libavutil/file.h>
}

namespace spdl::core::detail {

MemoryMappedFile::MemoryMappedFile(const std::string& path) {
  CHECK_AVERROR(
      av_file_map(path.data(), &buffer_, &buffer_size_, 0, NULL),
      "Failed to map file ({}).",
      path);
};

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept {
  *this = std::move(other);
}

MemoryMappedFile& MemoryMappedFile::operator=(
    MemoryMappedFile&& other) noexcept {
  using std::swap;
  swap(buffer_, other.buffer_);
  swap(buffer_size_, other.buffer_size_);
  swap(pos_, other.pos_);
  return *this;
}

MemoryMappedFile::~MemoryMappedFile() {
  av_file_unmap(buffer_, buffer_size_);
}

int MemoryMappedFile::read_packet(uint8_t* buf, int buf_size) {
  buf_size = FFMIN(buf_size, buffer_size_ - pos_);
  if (!buf_size) {
    return AVERROR_EOF;
  }
  memcpy(buf, buffer_ + pos_, buf_size);
  pos_ += buf_size;
  return buf_size;
}

int64_t MemoryMappedFile::seek(int64_t offset, int whence) {
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
}

int MemoryMappedFile::read_packet(void* opaque, uint8_t* buf, int buf_size) {
  return static_cast<MemoryMappedFile*>(opaque)->read_packet(buf, buf_size);
}

int64_t MemoryMappedFile::seek(void* opaque, int64_t offset, int whence) {
  return static_cast<MemoryMappedFile*>(opaque)->seek(offset, whence);
}

} // namespace spdl::core::detail
