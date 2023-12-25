#pragma once

#include <libspdl/defs.h>
#include <libspdl/detail/ffmpeg/wrappers.h>

#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace spdl::detail {

void clear_cuda_context_cache();

void create_cuda_context(
    const int index,
    const bool use_primary_context = false);

AVIOContextPtr get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence));

AVFormatInputContextPtr get_input_format_ctx(
    const std::string_view uri,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options);

AVFormatInputContextPtr get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options);

AVCodecContextPtr get_codec_ctx(
    const AVCodecParameters* params,
    AVRational pkt_timebase,
    const std::optional<std::string>& decoder = std::nullopt,
    const std::optional<OptionDict>& decoder_options = std::nullopt,
    const int cuda_device_index = -1);

} // namespace spdl::detail
