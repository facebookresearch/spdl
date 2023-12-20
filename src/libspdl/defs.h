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

} // namespace spdl

struct AVFrame;

namespace spdl {

// We deal with multiple frames at a time, so we use vector of raw
// pointers with dedicated destructor, as opposed to vector of managed pointers
// Each AVFrame* is expected to be created by av_read_frame.
// Therefore, they are reference counted and the counter should be 1.
// When destructing, they will be first unreferenced with av_frame_unref,
// then the data must be released with av_frame_free.
struct DecodedFrames {
  std::vector<AVFrame*> frames{};

  explicit DecodedFrames() = default;
  // No copy constructors
  DecodedFrames(const DecodedFrames&) = delete;
  DecodedFrames& operator=(const DecodedFrames&) = delete;
  // Move constructors to support MPMCQueue (BoundedQueue)
  DecodedFrames(DecodedFrames&&) noexcept = default;
  DecodedFrames& operator=(DecodedFrames&&) noexcept = default;
  // Destructor releases AVFrame* resources
  ~DecodedFrames();
};

// video buffer class to be exposed to python
struct VideoBuffer {
  size_t n, c, h, w;
  bool channel_last = false;
  std::vector<uint8_t> data;
};

} // namespace spdl
