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

} // namespace spdl
