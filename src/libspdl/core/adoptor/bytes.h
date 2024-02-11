#pragma once

#include <libspdl/core/adoptor/custom.h>

#include <string>

namespace spdl::core {
namespace detail {

class Bytes {
  std::vector<char> buffer{};
  int64_t pos = 0;

 public:
  Bytes(const std::string& path);

  Bytes(const Bytes&) = delete;
  Bytes& operator=(const Bytes&) = delete;

  Bytes(Bytes&&) noexcept;
  Bytes& operator=(Bytes&&) noexcept;

  ~Bytes() = default;

 private:
  int read_packet(uint8_t* buf, int buf_size);
  int64_t seek(int64_t offset, int whence);

 public:
  static int read_packet(void* opaque, uint8_t* buf, int buf_size);

  static int64_t seek(void* opaque, int64_t offset, int whence);
};

} // namespace detail

using BytesAdoptor = CustomAdoptor<detail::Bytes>;

} // namespace spdl::core
