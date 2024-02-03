#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

namespace spdl::core {

std::vector<std::unique_ptr<FrameContainer>> decode_video(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg);

std::vector<std::unique_ptr<FrameContainer>> decode_audio(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::string& filter_desc,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg);

} // namespace spdl::core
