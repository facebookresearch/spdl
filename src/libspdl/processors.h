#pragma once

#include <libspdl/frames.h>
#include <libspdl/types.h>
#include <vector>

namespace spdl {

struct DecodeConfig {
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
};

std::vector<Frames> decode_video(
    DecodeConfig cfg,
    const std::string& filter_desc);

} // namespace spdl
