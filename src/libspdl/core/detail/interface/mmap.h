#pragma once
#include <string>

namespace spdl::core::detail {

class MemoryMappedFile {
  uint8_t* buffer_ = nullptr;
  size_t buffer_size_ = 0;

  int64_t pos_ = 0;

 public:
  MemoryMappedFile(const std::string path);

  MemoryMappedFile(const MemoryMappedFile&) = delete;
  MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

  MemoryMappedFile(MemoryMappedFile&&) noexcept = default;

  ~MemoryMappedFile();

 private:
  int read_packet(uint8_t* buf, int buf_size);
  int64_t seek(int64_t offset, int whence);

 public:
  static int read_packet(void* opaque, uint8_t* buf, int buf_size);

  static int64_t seek(void* opaque, int64_t offset, int whence);
};

} // namespace spdl::core::detail
