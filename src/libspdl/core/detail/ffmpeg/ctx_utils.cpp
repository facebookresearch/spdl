#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <cassert>
#include <filesystem>
#include <mutex>
#include <set>

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
      assert(t);
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
// AVFormatContext (decode)
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
      TRACE_EVENT("decoding", "avcodec_find_decoder_by_name");
      auto c = avcodec_find_decoder_by_name(decoder_name->c_str());
      if (!c) {
        SPDL_FAIL(fmt::format("Unsupported codec: {}", decoder_name.value()));
      }
      return c;
    } else {
      TRACE_EVENT("decoding", "avcodec_parameters_to_context");
      auto c = avcodec_find_decoder(codec_id);
      if (!c) {
        SPDL_FAIL(
            fmt::format("Unsupported codec: {}", avcodec_get_name(codec_id)));
      }
      return c;
    }
  }();

  TRACE_EVENT("decoding", "avcodec_alloc_context3");
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

AVCodecContextPtr get_decode_codec_ctx_ptr(
    const AVCodecParameters* params,
    Rational pkt_timebase,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_options) {
  AVCodecContextPtr codec_ctx = alloc_codec_context(params->codec_id, decoder);

  VLOG(9) << "Configuring codec context.";
  {
    TRACE_EVENT("decoding", "avcodec_parameters_to_context");
    CHECK_AVERROR(
        avcodec_parameters_to_context(codec_ctx.get(), params),
        "Failed to set CodecContext parameter.");
  }
  VLOG(9) << "Codec: " << codec_ctx->codec->name;

  codec_ctx->pkt_timebase = AVRational{pkt_timebase.num, pkt_timebase.den};
  open_codec(codec_ctx.get(), decoder_options);
  return codec_ctx;
}

////////////////////////////////////////////////////////////////////////////////
// AVFormatContext (encode)
////////////////////////////////////////////////////////////////////////////////
AVFormatOutputContextPtr get_output_format_ctx(
    const std::string url,
    const std::optional<std::string>& format) {
  AVFormatContext* p = nullptr;
  CHECK_AVERROR(
      avformat_alloc_output_context2(
          &p, nullptr, format ? format->c_str() : nullptr, url.c_str()),
      fmt::format("Failed to allocate output format context for {}.", url));
  return AVFormatOutputContextPtr{p};
}

AVFormatOutputContextPtr get_output_format_ctx(
    AVIOContext* io_ctx,
    std::string format,
    const std::optional<std::string>& name) {
  AVFormatContext* p = nullptr;
  std::string nm = name ? *name : "Custom Output Context";
  CHECK_AVERROR(
      avformat_alloc_output_context2(&p, nullptr, format.c_str(), nm.c_str()),
      fmt::format("Failed to allocate output format context for {}.", nm));
  return AVFormatOutputContextPtr{p};
}

////////////////////////////////////////////////////////////////////////////////
// Encoding
////////////////////////////////////////////////////////////////////////////////
const AVCodec* get_image_codec(
    const std::optional<std::string>& encoder,
    const AVOutputFormat* oformat,
    const std::string& url) {
  if (encoder) {
    const AVCodec* c = avcodec_find_encoder_by_name(encoder->c_str());
    if (!c) {
      SPDL_FAIL(fmt::format("Unexpected encoder name: {}", encoder.value()));
    }
    return c;
  }

  // FFmpeg defaults to 'image2' muxer, of which default encoder is MJPEG.
  // This also applies to formats like PNG and TIFF
  if (std::strcmp(oformat->name, "image2") == 0) {
    // The following list was obtained by running
    // ffmpeg -h muxer=image2 | grep 'Common extensions'
    // then for each extension, checking the encoder by
    // fmpeg -hide_banner -h encoder=$ext | grep 'Encoder '
    static const std::set<std::string> exts{
        "bmp", "dpx",    "exr", "pam",   "pbm", "pcx", "pfm",
        "pgm", "pgmyuv", "phm", "png",   "ppm", "sgi", "tiff",
        "xwd", "vbn",    "xbm", "xface", "qoi", "hdr", "wbmp"};

    auto ext = std::filesystem::path(url).extension().string();
    if (!ext.empty()) {
      ext = ext.substr(1);
      if (exts.contains(ext)) {
        const AVCodec* c = avcodec_find_encoder_by_name(ext.c_str());
        if (c) {
          return c;
        }
      }
    }
  }

  auto default_codec = oformat->video_codec;

  const AVCodec* c = avcodec_find_encoder(default_codec);
  if (!c) {
    SPDL_FAIL(fmt::format(
        "Encoder not found for codec: {}", avcodec_get_name(default_codec)));
  }
  return c;
}

namespace {
bool is_pix_fmt_supported(
    const AVPixelFormat fmt,
    const AVPixelFormat* pix_fmts) {
  if (!pix_fmts) {
    return true;
  }
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    if (fmt == *pix_fmts) {
      return true;
    }
    ++pix_fmts;
  }
  return false;
}

std::string to_str(const AVPixelFormat* pix_fmts) {
  std::vector<std::string> ret;
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    ret.emplace_back(av_get_pix_fmt_name(*pix_fmts));
    ++pix_fmts;
  }
  return fmt::to_string(fmt::join(ret, ", "));
}
} // namespace

AVPixelFormat get_enc_fmt(
    AVPixelFormat src_fmt,
    const std::optional<std::string>& encoder_format,
    const AVCodec* codec) {
  if (encoder_format) {
    const auto& val = encoder_format.value();
    auto fmt = av_get_pix_fmt(val.c_str());
    if (!is_pix_fmt_supported(fmt, codec->pix_fmts)) {
      SPDL_FAIL(fmt::format(
          "Codec {} does not support {} format. Supported values are; {}",
          codec->name,
          val,
          to_str(codec->pix_fmts)));
    }
    return fmt;
  }
  if (codec->pix_fmts) {
    return codec->pix_fmts[0];
  }
  return src_fmt;
}

AVCodecContextPtr get_codec_ctx(const AVCodec* codec, int flags) {
  auto ctx = AVCodecContextPtr{CHECK_AVALLOCATE(avcodec_alloc_context3(codec))};

  if (flags & AVFMT_GLOBALHEADER) {
    ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  return ctx;
}

void configure_video_codec_ctx(
    AVCodecContextPtr& ctx,
    AVPixelFormat format,
    AVRational frame_rate,
    int width,
    int height,
    const EncodeConfig& cfg) {
  // TODO: Review other options and make them configurable?
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00147
  //  - bit_rate_tolerance
  //  - mb_decisions

  ctx->pix_fmt = format;
  ctx->width = width;
  ctx->height = height;
  ctx->time_base = av_inv_q(frame_rate);

  // Set optional stuff
  if (cfg.bit_rate > 0) {
    ctx->bit_rate = cfg.bit_rate;
  }
  if (cfg.compression_level != -1) {
    ctx->compression_level = cfg.compression_level;
  }
  if (cfg.gop_size != -1) {
    ctx->gop_size = cfg.gop_size;
  }
  if (cfg.max_b_frames != -1) {
    ctx->max_b_frames = cfg.max_b_frames;
  }
  if (cfg.qscale >= 0) {
    ctx->flags |= AV_CODEC_FLAG_QSCALE;
    ctx->global_quality = FF_QP2LAMBDA * cfg.qscale;
  }
}

void configure_image_codec_ctx(
    AVCodecContextPtr& ctx,
    AVPixelFormat format,
    int width,
    int height,
    const EncodeConfig& cfg) {
  configure_video_codec_ctx(ctx, format, {1, 1}, width, height, cfg);
}

template <MediaType media_type>
void open_codec(
    AVCodecContext* codec_ctx,
    const std::optional<OptionDict>& encode_options) {
  AVDictionaryDPtr option = get_option_dict(encode_options);

  if constexpr (media_type == MediaType::Audio) {
    // Enable experimental feature if required
    // Note:
    // "vorbis" refers to FFmpeg's native encoder,
    // https://ffmpeg.org/doxygen/4.1/vorbisenc_8c.html#a8c2e524b0f125f045fef39c747561450
    // while "libvorbis" refers to the one depends on libvorbis,
    // which is not experimental
    // https://ffmpeg.org/doxygen/4.1/libvorbisenc_8c.html#a5dd5fc671e2df9c5b1f97b2ee53d4025
    // similarly, "opus" refers to FFmpeg's native encoder
    // https://ffmpeg.org/doxygen/4.1/opusenc_8c.html#a05b203d4a9a231cc1fd5a7ddeb68cebc
    // while "libopus" refers to the one depends on libopusenc
    // https://ffmpeg.org/doxygen/4.1/libopusenc_8c.html#aa1d649e48cd2ec00cfe181cf9d0f3251
    if (std::strcmp(codec_ctx->codec->name, "vorbis") == 0) {
      if (!av_dict_get(option, "strict", nullptr, 0)) {
        LOG_FIRST_N(WARNING, 1)
            << "\"vorbis\" encoder is selected. Enabling '-strict experimental'. "
               "If this is not desired, please provide \"strict\" encoder option "
               "with desired value.";
        av_dict_set(option, "strict", "experimental", 0);
      }
    }
    if (std::strcmp(codec_ctx->codec->name, "opus") == 0) {
      if (!av_dict_get(option, "strict", nullptr, 0)) {
        LOG_FIRST_N(WARNING, 1)
            << "\"opus\" encoder is selected. Enabling '-strict experimental'. "
               "If this is not desired, please provide \"strict\" encoder option "
               "with desired value.";
        av_dict_set(option, "strict", "experimental", 0);
      }
    }
  }

  // Default to single thread execution.
  if (!av_dict_get(option, "threads", nullptr, 0)) {
    av_dict_set(option, "threads", "1", 0);
  }

  CHECK_AVERROR(
      avcodec_open2(codec_ctx, codec_ctx->codec, option),
      "Failed to open codec context.");
  check_empty(option);
}

template void open_codec<MediaType::Image>(
    AVCodecContext* codec_ctx,
    const std::optional<OptionDict>& encode_options);

AVStream* create_stream(
    AVFormatContext* format_ctx,
    AVCodecContext* codec_ctx) {
  AVStream* stream = CHECK_AVALLOCATE(avformat_new_stream(format_ctx, nullptr));
  // Note: time base may be adjusted when `calling avformat_write_header()`
  // https://ffmpeg.org/doxygen/5.1/structAVStream.html#a9db755451f14e2bf590d4b85d82b32e6
  stream->time_base = codec_ctx->time_base;
  CHECK_AVERROR(
      avcodec_parameters_from_context(stream->codecpar, codec_ctx),
      "Failed to create a new stream.");
  return stream;
}

void open_format(
    AVFormatContext* format_ctx,
    const std::optional<OptionDict>& options) {
  AVFORMAT_CONST AVOutputFormat* fmt = format_ctx->oformat;
  AVDictionaryDPtr option = get_option_dict(options);

  if (strcmp(fmt->name, "image2") == 0) {
    // By default, image2 muxer warns about the path not containing sequence
    // number everytime the codec is initialized. For the case of single image
    // encoding, this is unnecessary and super annoying. So we set the update
    // flag to 1.
    // https://github.com/FFmpeg/FFmpeg/blob/e757726e89ff636e0dc6743f635888639a196e36/libavformat/img2enc.c#L171-L174
    if (!av_dict_get(option, "update", nullptr, 0)) {
      av_dict_set(option, "update", "1", 0);
    }
  }

  if (!(fmt->flags & AVFMT_NOFILE) &&
      !(format_ctx->flags & AVFMT_FLAG_CUSTOM_IO)) {
    CHECK_AVERROR(
        avio_open2(
            &format_ctx->pb, format_ctx->url, AVIO_FLAG_WRITE, nullptr, option),
        fmt::format("Failed to open custom output: {}", format_ctx->url));
  }

  CHECK_AVERROR(
      avformat_write_header(format_ctx, option),
      fmt::format("Failed to write header: {}", format_ctx->url));
  check_empty(option);
}

} // namespace spdl::core::detail
