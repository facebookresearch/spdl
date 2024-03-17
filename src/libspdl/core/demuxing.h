#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/future.h>
#include <libspdl/core/packets.h>
#include <libspdl/core/types.h>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Demuxing
////////////////////////////////////////////////////////////////////////////////

/// Demux audio or video
template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(std::optional<PacketsPtr<media_type>>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

/// Demux single image
FuturePtr async_demux_image(
    std::function<void(std::optional<ImagePacketsPtr>)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr demux_executor);

/// Apply bit stream filtering for NVDEC decoding
FuturePtr async_apply_bsf(
    std::function<void(VideoPacketsPtr)> set_result,
    std::function<void()> notify_exception,
    VideoPacketsPtr packets,
    ThreadPoolExecutorPtr executor);

} // namespace spdl::core
