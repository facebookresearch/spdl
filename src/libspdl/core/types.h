#pragma once

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace spdl::core {

using OptionDict = std::map<std::string, std::string>;

// alternative for AVRational so that we can avoid exposing FFmpeg headers
using Rational = std::tuple<int, int>;

// simplified version of AVMediaType so that public headers do not
// include ffmpeg headers
enum class MediaType { Audio, Video };

struct IOConfig {
  std::optional<std::string> format = std::nullopt;
  std::optional<OptionDict> format_options = std::nullopt;
  int buffer_size = 8096;
};

struct DecodeConfig {
  std::optional<std::string> decoder = std::nullopt;
  std::optional<OptionDict> decoder_options = std::nullopt;
  int cuda_device_index = -1;
};

// Used to construct Dtype when converting buffer to array
enum class ElemClass { Int, UInt, Float };

} // namespace spdl::core
