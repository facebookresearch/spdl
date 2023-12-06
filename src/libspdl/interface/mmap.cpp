extern "C" {
#include <libavformat/avio.h>
#include <libavutil/file.h>
}

#include <string>

#include <libspdl/ffmpeg/logging.h>
#include <libspdl/interface/mmap.h>

namespace spdl::interface {

MemoryMappedFile::MemoryMappedFile(const std::string_view path) : path_(path) {
  CHECK_AVERROR(
      av_file_map(path_.c_str(), &buffer_, &buffer_size_, 0, NULL),
      "Failed to map file ({}).",
      path);
};

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
      LOG(ERROR) << "Unexpected whence value was found: " << whence;
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

} // namespace spdl::interface
