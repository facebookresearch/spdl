#pragma once

#include <libspdl/core/adoptor.h>
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

/// Demux audio or video from source URI
template <MediaType media_type>
FuturePtr async_demux(
    std::function<void(std::optional<PacketsWrapperPtr<media_type>>)>
        set_result,
    std::function<void()> notify_exception,
    const std::string& uri,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor);

/// Demux audio or video from byte string stored somewhere.
///
/// Note: It is caller's responsibility to keep the data alive until the
/// returned Future is destroyed.
template <MediaType media_type>
FuturePtr async_demux_bytes(
    std::function<void(std::optional<PacketsWrapperPtr<media_type>>)>
        set_result,
    std::function<void()> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor);

/// Demux single image from source URI
FuturePtr async_demux_image(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void()> notify_exception,
    const std::string& src,
    const SourceAdoptorPtr& adoptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor);

/// Demux image from byte string stored somewhere.
///
/// Note: It is caller's responsibility to keep the data alive until the
/// returned Future is destroyed.
FuturePtr async_demux_image_bytes(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void()> notify_exception,
    std::string_view data,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor);

/// Apply bit stream filtering for NVDEC decoding
FuturePtr async_apply_bsf(
    std::function<void(VideoPacketsWrapperPtr)> set_result,
    std::function<void()> notify_exception,
    VideoPacketsWrapperPtr packets,
    ThreadPoolExecutorPtr executor);

} // namespace spdl::core
