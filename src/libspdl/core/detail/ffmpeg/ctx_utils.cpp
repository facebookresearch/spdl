#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

#include <mutex>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/hwcontext.h>
}

// https://github.com/FFmpeg/FFmpeg/blob/4e6debe1df7d53f3f59b37449b82265d5c08a172/doc/APIchanges#L252-L260
// Starting from libavformat 59 (ffmpeg 5),
// AVInputFormat is const and related functions expect constant.
#if LIBAVFORMAT_VERSION_MAJOR >= 59
#define AVFORMAT_CONST const
#else
#define AVFORMAT_CONST
#endif

namespace spdl::core::detail {
namespace {
//////////////////////////////////////////////////////////////////////////////
// AVDictionary
//////////////////////////////////////////////////////////////////////////////

DEF_DPtr(AVDictionary, av_dict_free); // This defines AVDictionaryDPtr class

AVDictionaryDPtr get_option_dict(const std::optional<OptionDict>& options) {
  AVDictionaryDPtr opt;
  if (options) {
    for (const auto& [key, value] : options.value()) {
      CHECK_AVERROR(
          av_dict_set(opt, key.c_str(), value.c_str(), 0),
          "Failed to convert option dictionary. ({}={})",
          key,
          value);
    }
  }
  return opt;
}

void check_empty(const AVDictionary* p) {
  if (av_dict_count(p)) {
    AVDictionaryEntry* t = nullptr;
    std::vector<std::string> keys;
    while ((t = av_dict_get(p, "", t, AV_DICT_IGNORE_SUFFIX))) {
      keys.emplace_back(t->key);
    }
    SPDL_FAIL(fmt::format("Unexpected options: {}", fmt::join(keys, ", ")));
  }
}
} // namespace

////////////////////////////////////////////////////////////////////////////////
// AVIOContext
////////////////////////////////////////////////////////////////////////////////
AVIOContextPtr get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence)) {
  auto buffer =
      static_cast<unsigned char*>(CHECK_AVALLOCATE(av_malloc(buffer_size)));
  AVIOContextPtr io_ctx;
  {
    TRACE_EVENT("decoding", "avio_alloc_context");
    io_ctx.reset(avio_alloc_context(
        buffer, buffer_size, 0, opaque, read_packet, nullptr, seek));
  }
  if (!io_ctx) [[unlikely]] {
    av_freep(&buffer);
    SPDL_FAIL("Failed to allocate AVIOContext.");
  }
  return io_ctx;
}

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext
////////////////////////////////////////////////////////////////////////////////
namespace {
AVFormatInputContextPtr get_input_format_ctx(
    const char* src,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options,
    AVIOContext* io_ctx) {
  // We check the input format first because the heap data is owned by FFmpeg
  // library, so we do't need to free it in case of an error.
  auto in_fmt = [&format]() {
    AVFORMAT_CONST AVInputFormat* fmt = nullptr;
    if (format) {
      fmt = av_find_input_format(format->c_str());
      if (!fmt) [[unlikely]] {
        SPDL_FAIL(fmt::format("Unsupported device/format: {}", format.value()));
      }
    }
    return fmt;
  }();

  AVDictionaryDPtr option = get_option_dict(format_options);

  // Note:
  // If `avformat_open_input` fails, it frees fmt_ctx.
  // So we use raw pointer untill we know `avformat_open_input` succeeded.
  // https://ffmpeg.org/doxygen/5.0/group__lavf__decoding.html#gac05d61a2b492ae3985c658f34622c19d
  AVFormatContext* fmt_ctx = CHECK_AVALLOCATE(avformat_alloc_context());
  if (io_ctx) {
    fmt_ctx->pb = io_ctx;
  }
  int errnum;
  {
    TRACE_EVENT("decoding", "avformat_open_input");
    errnum = avformat_open_input(&fmt_ctx, src, in_fmt, option);
  }
  if (errnum < 0) [[unlikely]] {
    SPDL_FAIL(
        src ? av_error(errnum, "Failed to open the input: {}", src)
            : av_error(errnum, "Failed to open custom input."));
  }
  // Now pass down the responsibility of resource clean up to RAII.
  AVFormatInputContextPtr ret{fmt_ctx};

  check_empty(option);
  return ret;
}

} // namespace

AVFormatInputContextPtr get_input_format_ctx(
    const std::string url,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options) {
  return get_input_format_ctx(url.data(), format, format_options, nullptr);
}

AVFormatInputContextPtr get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options) {
  return get_input_format_ctx(nullptr, format, format_options, io_ctx);
}

//////////////////////////////////////////////////////////////////////////////
// AVCodecContext
//////////////////////////////////////////////////////////////////////////////
namespace {
AVCodecContextPtr alloc_codec_context(
    enum AVCodecID codec_id,
    const std::optional<std::string>& decoder_name) {
  auto codec = [&]() -> const AVCodec* {
    if (decoder_name) {
      auto c = avcodec_find_decoder_by_name(decoder_name->c_str());
      if (!c) {
        SPDL_FAIL(fmt::format("Unsupported codec: {}", decoder_name.value()));
      }
      return c;
    } else {
      auto c = avcodec_find_decoder(codec_id);
      if (!c) {
        SPDL_FAIL(
            fmt::format("Unsupported codec: {}", avcodec_get_name(codec_id)));
      }
      return c;
    }
  }();

  auto* codec_ctx = CHECK_AVALLOCATE(avcodec_alloc_context3(codec));
  return AVCodecContextPtr{codec_ctx};
}

void open_codec(
    AVCodecContext* codec_ctx,
    const std::optional<OptionDict>& decoder_options) {
  AVDictionaryDPtr option = get_option_dict(decoder_options);

  // Default to single thread execution.
  if (!av_dict_get(option, "threads", nullptr, 0)) {
    av_dict_set(option, "threads", "1", 0);
  }
  {
    TRACE_EVENT("decoding", "avcodec_open2");
    CHECK_AVERROR(
        avcodec_open2(codec_ctx, codec_ctx->codec, option),
        "Failed to initialize CodecContext.");
  }
  check_empty(option);
}

} // namespace

AVCodecContextPtr get_codec_ctx_ptr(
    const AVCodecParameters* params,
    AVRational pkt_timebase,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_options) {
  AVCodecContextPtr codec_ctx = alloc_codec_context(params->codec_id, decoder);

  XLOG(DBG9) << "Configuring codec context.";
  CHECK_AVERROR(
      avcodec_parameters_to_context(codec_ctx.get(), params),
      "Failed to set CodecContext parameter.");
  XLOG(DBG9) << "Codec: " << codec_ctx->codec->name;

  codec_ctx->pkt_timebase = pkt_timebase;
  open_codec(codec_ctx.get(), decoder_options);
  return codec_ctx;
}

} // namespace spdl::core::detail
