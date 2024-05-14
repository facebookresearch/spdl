#pragma once

#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define SPDL_DEFAULT_BUFFER_SIZE 8096

namespace spdl::core {

using OptionDict = std::map<std::string, std::string>;

// alternative for AVRational so that we can avoid exposing FFmpeg headers
struct Rational {
  int num = 0;
  int den = 1;
};

// simplified version of AVMediaType so that public headers do not
// include ffmpeg headers
enum class MediaType { Audio, Video, Image };

struct DemuxConfig {
  std::optional<std::string> format = std::nullopt;
  std::optional<OptionDict> format_options = std::nullopt;
  int buffer_size = SPDL_DEFAULT_BUFFER_SIZE;
};

struct DecodeConfig {
  std::optional<std::string> decoder = std::nullopt;
  std::optional<OptionDict> decoder_options = std::nullopt;
};

// Used to construct Dtype when converting buffer to array
enum class ElemClass { Int, UInt, Float };

struct CropArea {
  short left = 0;
  short top = 0;
  short right = 0;
  short bottom = 0;
};

struct EncodeConfig {
  // Muxer format or device name
  std::optional<std::string> muxer = std::nullopt;
  // Options for muxer
  std::optional<OptionDict> muxer_options = std::nullopt;

  // Encoder
  std::optional<std::string> encoder = std::nullopt;
  // Encoder options
  std::optional<OptionDict> encoder_options = std::nullopt;

  // Pixel/sample format used for encoding
  std::optional<std::string> format = std::nullopt;

  // Rescale options
  int width = -1;
  int height = -1;
  std::optional<std::string> scale_algo = std::nullopt;

  // Optional filter desc
  std::optional<std::string> filter_desc = std::nullopt;

  int bit_rate = -1;
  int compression_level = -1;

  // qscale corresponds to ffmpeg CLI's qscale.
  // Example: MP3
  // https://trac.ffmpeg.org/wiki/Encode/MP3
  // This should be set like
  // https://github.com/FFmpeg/FFmpeg/blob/n4.3.2/fftools/ffmpeg_opt.c#L1550
  int qscale = -1;

  // video
  int gop_size = -1;
  int max_b_frames = -1;
};

// Thrown when unexpected internal error occurs.
class InternalError : public std::logic_error {
  using std::logic_error::logic_error;
};

} // namespace spdl::core
