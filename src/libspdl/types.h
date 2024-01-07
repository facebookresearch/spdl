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

// simplified version of AVMediaType so that public headers do not
// include ffmpeg headers
enum class MediaType { NA, Audio, Video };

// Used to construct Dtype when converting buffer to array
enum class ElemClass { Int, UInt, Float };

} // namespace spdl
