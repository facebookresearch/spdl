#include <libspdl/core/adoptor/bytes.h>

#include <libspdl/core/logging.h>

#include <folly/logging/xlog.h>

#include <fstream>

extern "C" {
#include <libavformat/avio.h>
#include <libavutil/error.h>
}

namespace spdl::core::detail {
namespace {
std::vector<char> read(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (ifs.fail()) {
    SPDL_FAIL(fmt::format("Failed to open file: {}", filename));
  }
  std::ifstream::pos_type pos = ifs.tellg();

  if (pos == 0) {
    return std::vector<char>{};
  }

  std::vector<char> ret(pos);
  ifs.seekg(0, std::ios::beg);
  ifs.read(&ret[0], pos);
  return ret;
}
} // namespace

Bytes::Bytes(const std::string& path) : buffer(read(path)) {
  XLOG(DBG5) << fmt::format("Loaded {} bytes from {}.", buffer.size(), path);
}

Bytes::Bytes(Bytes&& other) noexcept {
  *this = std::move(other);
}

Bytes& Bytes::operator=(Bytes&& other) noexcept {
  using std::swap;
  swap(buffer, other.buffer);
  swap(pos, other.pos);
  return *this;
}

int Bytes::read_packet(uint8_t* buf, int buf_size) {
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

int64_t Bytes::seek(int64_t offset, int whence) {
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

int Bytes::read_packet(void* opaque, uint8_t* buf, int buf_size) {
  return static_cast<Bytes*>(opaque)->read_packet(buf, buf_size);
}

int64_t Bytes::seek(void* opaque, int64_t offset, int whence) {
  return static_cast<Bytes*>(opaque)->seek(offset, whence);
}

} // namespace spdl::core::detail
