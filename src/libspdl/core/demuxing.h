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
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    std::vector<std::tuple<double, double>> timestamps,
    SourceAdaptorPtr adaptor = nullptr,
    std::optional<IOConfig> io_cfg = std::nullopt,
    ThreadPoolExecutorPtr executor = nullptr);

/// Demux audio or video from byte string stored somewhere.
///
/// Note: It is caller's responsibility to keep the data alive until the
/// returned Future is destroyed.
template <MediaType media_type>
FuturePtr async_demux_bytes(
    std::function<void(PacketsPtr<media_type>)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    std::vector<std::tuple<double, double>> timestamps,
    std::optional<IOConfig> io_cfg = std::nullopt,
    ThreadPoolExecutorPtr executor = nullptr,
    bool _zero_clear = false);

/// Demux single image from source URI
FuturePtr async_demux_image(
    std::function<void(ImagePacketsPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string uri,
    SourceAdaptorPtr adaptor = nullptr,
    std::optional<IOConfig> io_cfg = std::nullopt,
    ThreadPoolExecutorPtr executor = nullptr);

/// Demux image from byte string stored somewhere.
///
/// Note: It is caller's responsibility to keep the data alive until the
/// returned Future is destroyed.
FuturePtr async_demux_image_bytes(
    std::function<void(ImagePacketsPtr)> set_result,
    std::function<void(std::string, bool)> notify_exception,
    std::string_view data,
    std::optional<IOConfig> io_cfg = std::nullopt,
    ThreadPoolExecutorPtr executor = nullptr,
    bool _zero_clear = false);

} // namespace spdl::core
