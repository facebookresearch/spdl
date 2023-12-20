#pragma once

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace spdl {

using OptionDict = std::map<std::string, std::string>;

// alternative for AVRational so that we can avoid exposing FFmpeg headers
using Rational = std::tuple<int, int>;

struct VideoDecodingJob {
  std::string src;
  std::vector<double> timestamps;

  // I/O config
  std::optional<std::string> format = std::nullopt;
  std::optional<OptionDict> format_options = std::nullopt;
  int buffer_size = 8096;

  // decoder config
  std::optional<std::string> decoder = std::nullopt;
  std::optional<OptionDict> decoder_options = std::nullopt;
  int cuda_device_index = -1;

  // Post processing config
  std::optional<Rational> frame_rate = std::nullopt;
  std::optional<int> width = std::nullopt;
  std::optional<int> height = std::nullopt;
  std::optional<std::string> pix_fmt = std::nullopt;
};

// video buffer class to be exposed to python
struct VideoBuffer {
  size_t n, c, h, w;
  bool channel_last = false;
  std::vector<uint8_t> data;
};

} // namespace spdl
