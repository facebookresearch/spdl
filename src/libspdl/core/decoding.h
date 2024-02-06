#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/interface/base.h>
#include <libspdl/core/types.h>

#include <memory>
#include <vector>

namespace spdl::core {

std::vector<std::unique_ptr<FrameContainer>> decode_video(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc);

std::vector<std::unique_ptr<FrameContainer>> decode_audio(
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const std::shared_ptr<SourceAdoptor>& adoptor,
    const IOConfig& io_cfg,
    const DecodeConfig& decode_cfg,
    const std::string& filter_desc);

} // namespace spdl::core
