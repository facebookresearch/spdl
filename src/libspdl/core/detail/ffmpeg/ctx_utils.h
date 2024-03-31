#pragma once

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace spdl::core::detail {

void clear_cuda_context_cache();

int get_device_index_from_frame_context(const AVBufferRef* hw_frames_ctx);

void create_cuda_context(
    const int index,
    const bool use_primary_context = false);

AVIOContextPtr get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence));

AVFormatInputContextPtr get_input_format_ctx(
    const std::string url,
    const std::optional<std::string>& format = std::nullopt,
    const std::optional<OptionDict>& format_options = std::nullopt);

AVFormatInputContextPtr get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options);

AVCodecContextPtr get_codec_ctx_ptr(
    const AVCodecParameters* params,
    AVRational pkt_timebase,
    const std::optional<std::string>& decoder = std::nullopt,
    const std::optional<OptionDict>& decoder_options = std::nullopt,
    const int cuda_device_index = -1);

} // namespace spdl::core::detail
