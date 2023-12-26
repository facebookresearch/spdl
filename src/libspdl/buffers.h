#pragma once

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace spdl {

// video buffer class to be exposed to python
struct VideoBuffer {
  struct CPUStorage {
    std::unique_ptr<uint8_t[]> data;

    CPUStorage(size_t size);
  };

  std::vector<size_t> shape;
  bool channel_last = false;
  std::variant<CPUStorage> storage;

  VideoBuffer(
      const std::vector<size_t> shape,
      bool channel_last,
      CPUStorage storage);
  void* data();
};

VideoBuffer video_buffer(
    const std::vector<size_t> shape,
    bool channel_last = false);

} // namespace spdl
