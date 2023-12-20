#pragma once

#include <cstdint>
#include <vector>

namespace spdl {

// video buffer class to be exposed to python
struct VideoBuffer {
  size_t n = 0, c = 0, h = 0, w = 0;
  bool channel_last = false;
  std::vector<uint8_t> data{};
};

} // namespace spdl
