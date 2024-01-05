#pragma once

#include <libspdl/frames.h>
#include <libspdl/types.h>
#include <vector>

namespace spdl {

struct IOConfig {
  // I/O config
  std::optional<std::string> format = std::nullopt;
  std::optional<OptionDict> format_options = std::nullopt;
  int buffer_size = 8096;
};

struct DecodeConfig {
  // decoder config
  std::optional<std::string> decoder = std::nullopt;
  std::optional<OptionDict> decoder_options = std::nullopt;
  int cuda_device_index = -1;
};

std::vector<Frames> decode_video(
    const std::string& src,
    const std::vector<double>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg);

} // namespace spdl
