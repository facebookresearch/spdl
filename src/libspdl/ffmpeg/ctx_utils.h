#pragma once
#include <libspdl/ffmpeg/wrappers.h>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace spdl {

// https://github.com/FFmpeg/FFmpeg/blob/4e6debe1df7d53f3f59b37449b82265d5c08a172/doc/APIchanges#L252-L260
// Starting from libavformat 59 (ffmpeg 5),
// AVInputFormat is const and related functions expect constant.
#if LIBAVFORMAT_VERSION_MAJOR >= 59
#define AVFORMAT_CONST const
#else
#define AVFORMAT_CONST
#endif

using OptionDict = std::map<std::string, std::string>;

AVIOContextPtr get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence));

AVFormatInputContextPtr get_input_format_ctx(
    const std::string_view uri,
    const std::optional<OptionDict>& options,
    const std::optional<std::string>& format);

AVFormatInputContextPtr get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<OptionDict>& options,
    const std::optional<std::string>& format);

AVCodecContextPtr get_codec_ctx(
    const AVCodecParameters* params,
    AVRational pkt_timebase,
    const std::optional<std::string>& decoder_name = std::nullopt,
    const std::optional<OptionDict>& decoder_option = std::nullopt,
    const int cuda_device_index = -1);
} // namespace spdl
