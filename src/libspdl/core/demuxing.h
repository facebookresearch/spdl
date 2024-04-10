#pragma once

#include <libspdl/core/adaptor.h>
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
    std::function<void(PacketsWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& uri,
    const std::vector<std::tuple<double, double>>& timestamps,
    const SourceAdaptorPtr& adaptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor);

/// Demux audio or video from byte string stored somewhere.
///
/// Note: It is caller's responsibility to keep the data alive until the
/// returned Future is destroyed.
template <MediaType media_type>
FuturePtr async_demux_bytes(
    std::function<void(PacketsWrapperPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const std::vector<std::tuple<double, double>>& timestamps,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear = false);

/// Demux single image from source URI
FuturePtr async_demux_image(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    const std::string& src,
    const SourceAdaptorPtr& adaptor,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor);

/// Demux image from byte string stored somewhere.
///
/// Note: It is caller's responsibility to keep the data alive until the
/// returned Future is destroyed.
FuturePtr async_demux_image_bytes(
    std::function<void(ImagePacketsWrapperPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    const IOConfig& io_cfg,
    ThreadPoolExecutorPtr executor,
    bool _zero_clear = false);

} // namespace spdl::core
