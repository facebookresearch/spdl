#pragma once

#include <map>
#include <string>
#include <tuple>

namespace spdl {

using OptionDict = std::map<std::string, std::string>;
// alternative for AVRational so that we can avoid exposing FFmpeg headers
using Rational = std::tuple<int, int>;
} // namespace spdl
