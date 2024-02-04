#include <libspdl/core/detail/ffmpeg/ctx_utils.h>
#include <libspdl/core/detail/ffmpeg/logging.h>

#include <folly/logging/xlog.h>

#ifdef SPDL_USE_CUDA
#include <cuda.h>
#endif

#include <mutex>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/hwcontext.h>
#ifdef SPDL_USE_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
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
  AVDictionaryEntry* t = nullptr;
  if (p && av_dict_get(p, "", t, AV_DICT_IGNORE_SUFFIX)) [[unlikely]] {
    std::vector<std::string> keys{t->key};
    while ((t = av_dict_get(p, "", t, AV_DICT_IGNORE_SUFFIX))) {
      keys.emplace_back(t->key);
    }
    SPDL_FAIL(fmt::format("Unexpected options: {}", fmt::join(keys, ", ")));
  }
}
} // namespace

////////////////////////////////////////////////////////////////////////////////
// Hardware context
////////////////////////////////////////////////////////////////////////////////

namespace {

static std::mutex MUTEX;
static std::map<int, AVBufferRefPtr> CUDA_CONTEXT_CACHE;

} // namespace

#ifdef SPDL_USE_CUDA
void warn_if_device_is_initialized(int index) {
  // FFmpeg rejects the use of the primary context if it is already initialized
  // with undesired flags. i.e. FFmpeg only accepts CU_CTX_SCHED_BLOCKING_SYNC.
  //
  // PyTorch initializes the primary context without any flag.
  // So, the primary context must be created by FFmpeg before the client code
  // call any PyTorch utility that initializes the primary context.
  // This is out of our controll, so this function emits the warning before
  // the subsequent procedure fails.
  CUdevice device;
  auto result = cuDeviceGet(&device, index);
  if (result != CUDA_SUCCESS) {
    return;
  }

  int active = 0;
  unsigned int flags = 0;
  result = cuDevicePrimaryCtxGetState(device, &flags, &active);
  XLOG(DBG9) << fmt::format(
      "Device {}: flag={} ({}active)", index, flags, active ? "" : "in");
  if (result == CUDA_SUCCESS && active && flags != CU_CTX_SCHED_BLOCKING_SYNC) {
    XLOG(WARN) << fmt::format(
        "The primary CUDA device context is already initialized on device {} with flag {}. This likely causes a failure in subsequent decoder intialization procedures as FFmpeg rejects using primary context when the flag is not CU_CTX_SCHED_BLOCKING_SYNC. PyTorch initializes the primary context with incompatible flag by default (and there seems to be no way to configure it according to https://github.com/pytorch/pytorch/issues/28224. Plese initialize the CUDA context with proper flag by calling this function first before calling PyTorch functions.",
        index,
        flags);
  }
}
#endif

void create_cuda_context(const int index, const bool use_primary_context) {
#ifndef SPDL_USE_CUDA
  SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
  if (CUDA_CONTEXT_CACHE.count(index)) {
    SPDL_FAIL(fmt::format(
        "CUDA context for device {} is already initialized", index));
  }
  AVBufferRef* p = nullptr;
  AVDictionaryDPtr opt;
  unsigned int device_flags = 0;
  if (use_primary_context) {
    XLOG(DBG9) << "Creating CUDA context for device " << index
               << " using the primary context";
    warn_if_device_is_initialized(index);
    av_dict_set(opt, "primary_ctx", "1", 0);
    device_flags = CU_CTX_SCHED_BLOCKING_SYNC;
  }

  CHECK_AVERROR(
      av_hwdevice_ctx_create(
          &p,
          AV_HWDEVICE_TYPE_CUDA,
          std::to_string(index).c_str(),
          opt,
          device_flags),
      "Failed to create CUDA device context on device {}.",
      index);
  assert(p);
  CUDA_CONTEXT_CACHE.emplace(index, p);

  auto hw_device_ctx = (AVHWDeviceContext*)p->data;
  auto device_hwctx = (AVCUDADeviceContext*)hw_device_ctx->hwctx;
  XLOG(DBG9) << "CUcontext: " << device_hwctx->cuda_ctx;
  XLOG(DBG9) << "CUstream: " << device_hwctx->stream;
#endif
}

int get_device_index_from_frame_context(const AVBufferRef* hw_frames_ctx) {
  auto hw_frames_ctx_ = (AVHWFramesContext*)hw_frames_ctx->data;
  auto hw_device_ctx = (AVHWDeviceContext*)hw_frames_ctx_->device_ctx;

  for (auto const& [index, ref] : CUDA_CONTEXT_CACHE) {
    AVBufferRef* p = ref.get();
    auto ctx = (AVHWDeviceContext*)p->data;
    if (ctx == hw_device_ctx) {
      return index;
    }
  }
  SPDL_FAIL_INTERNAL(fmt::format(
      "Failed to locate the device associated with context {}",
      (void*)hw_device_ctx));
}

namespace {

AVBufferRef* get_cuda_context(int index, bool use_primary_context = false) {
  std::lock_guard<std::mutex> lock(MUTEX);
  if (index == -1) {
    index = 0;
  }
  if (CUDA_CONTEXT_CACHE.count(index) == 0) {
    create_cuda_context(index, use_primary_context);
  }
  AVBufferRefPtr& buffer = CUDA_CONTEXT_CACHE.at(index);
  return buffer.get();
}

} // namespace

void clear_cuda_context_cache() {
  XLOG(DBG9) << "Clearing CUDA context cache.";
  std::lock_guard<std::mutex> lock(MUTEX);
  CUDA_CONTEXT_CACHE.clear();
}

////////////////////////////////////////////////////////////////////////////////
// AVIOContext
////////////////////////////////////////////////////////////////////////////////
AVIOContext* get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence)) {
  auto buffer =
      static_cast<unsigned char*>(CHECK_AVALLOCATE(av_malloc(buffer_size)));
  AVIOContext* io_ctx = avio_alloc_context(
      buffer, buffer_size, 0, opaque, read_packet, nullptr, seek);
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
      fmt = av_find_input_format(format.value().c_str());
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
  int errnum = avformat_open_input(&fmt_ctx, src, in_fmt, option);
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

AVFormatInputContextPtr get_input_format_ctx_ptr(
    const std::string_view id,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options) {
  return get_input_format_ctx(id.data(), format, format_options, nullptr);
}

AVFormatContext* get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options) {
  return get_input_format_ctx(nullptr, format, format_options, io_ctx)
      .release();
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
      auto c = avcodec_find_decoder_by_name(decoder_name.value().c_str());
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

#ifdef SPDL_USE_CUDA
const AVCodecHWConfig* get_cuda_config(const AVCodec* codec) {
  for (int i = 0;; ++i) {
    const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
    if (!config) {
      break;
    }
    if (config->device_type == AV_HWDEVICE_TYPE_CUDA &&
        config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
      return config;
    }
  }
  SPDL_FAIL(fmt::format(
      "CUDA device was requested, but the codec \"{}\" is not supported.",
      codec->name));
}

enum AVPixelFormat get_hw_format(
    AVCodecContext* codec_ctx,
    const enum AVPixelFormat* pix_fmts) {
  const AVCodecHWConfig* cfg = static_cast<AVCodecHWConfig*>(codec_ctx->opaque);
  for (const enum AVPixelFormat* p = pix_fmts; *p != -1; p++) {
    if (*p == cfg->pix_fmt) {
      // Note
      // The HW decode example uses generic approach
      // https://ffmpeg.org/doxygen/4.1/hw__decode_8c_source.html#l00063
      // But this approach finalizes the codec configuration when the first
      // frame comes in.
      // We need to inspect the codec configuration right after the codec is
      // opened.
      // So we add short cut for known patterns.
      // yuv420p (h264) -> nv12
      // yuv420p10le (hevc/h265) -> p010le
      switch (codec_ctx->pix_fmt) {
        case AV_PIX_FMT_YUV420P: {
          codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
          codec_ctx->sw_pix_fmt = AV_PIX_FMT_NV12;
          break;
        }
        case AV_PIX_FMT_YUV420P10LE: {
          codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
          codec_ctx->sw_pix_fmt = AV_PIX_FMT_P010LE;
          break;
        }
        default:;
      }
      return *p;
    }
  }
  XLOG(WARN) << "Failed to get HW surface format.";
  return AV_PIX_FMT_NONE;
}

AVBufferRef* get_hw_frames_ctx(AVCodecContext* codec_ctx) {
  AVBufferRef* p = av_hwframe_ctx_alloc(codec_ctx->hw_device_ctx);
  if (!p) {
    SPDL_FAIL(fmt::format(
        "Failed to allocate CUDA frame context from device context at {}",
        fmt::ptr(codec_ctx->hw_device_ctx)));
  }
  auto frames_ctx = (AVHWFramesContext*)(p->data);
  frames_ctx->format = codec_ctx->pix_fmt;
  frames_ctx->sw_format = codec_ctx->sw_pix_fmt;
  frames_ctx->width = codec_ctx->width;
  frames_ctx->height = codec_ctx->height;
  frames_ctx->initial_pool_size = 5;
  int ret = av_hwframe_ctx_init(p);
  if (ret < 0) {
    av_buffer_unref(&p);
    SPDL_FAIL(av_error(ret, "Failed to initialize CUDA frame context."));
  }
  return p;
}
#endif

void configure_codec_context(
    AVCodecContext* codec_ctx,
    const AVCodecParameters* params,
    const int cuda_device_index) {
  XLOG(DBG9) << "Configuring codec context.";
  CHECK_AVERROR(
      avcodec_parameters_to_context(codec_ctx, params),
      "Failed to set CodecContext parameter.");
  XLOG(DBG9) << "Codec: " << codec_ctx->codec->name;

  if (!codec_ctx->channel_layout) {
    codec_ctx->channel_layout =
        av_get_default_channel_layout(codec_ctx->channels);
  }

  if (cuda_device_index >= 0) {
#ifndef SPDL_USE_CUDA
    SPDL_FAIL("SPDL is not compiled with CUDA support.");
#else
    XLOG(DBG9) << "Enabling CUDA acceleration";
    const AVCodecHWConfig* cfg = get_cuda_config(codec_ctx->codec);
    // https://www.ffmpeg.org/doxygen/trunk/hw__decode_8c_source.html#l00221
    // 1. Set HW config to opaue pointer.
    codec_ctx->opaque = static_cast<void*>(const_cast<AVCodecHWConfig*>(cfg));
    // 2. Set pCodecContext->get_format call back function which
    // will retrieve the HW pixel format from opaque pointer.
    codec_ctx->get_format = get_hw_format;
    codec_ctx->hw_device_ctx =
        av_buffer_ref(get_cuda_context(cuda_device_index));
    if (!codec_ctx->hw_device_ctx) {
      SPDL_FAIL("Failed to reference HW device context.");
    }
#endif
  }
}

void open_codec(
    AVCodecContext* codec_ctx,
    const std::optional<OptionDict>& decoder_options) {
  AVDictionaryDPtr option = get_option_dict(decoder_options);

  // Default to single thread execution.
  if (!av_dict_get(option, "threads", nullptr, 0)) {
    av_dict_set(option, "threads", "1", 0);
  }
  CHECK_AVERROR(
      avcodec_open2(codec_ctx, codec_ctx->codec, option),
      "Failed to initialize CodecContext.");
  check_empty(option);
}

} // namespace

AVCodecContextPtr get_codec_ctx_ptr(
    const AVCodecParameters* params,
    AVRational pkt_timebase,
    const std::optional<std::string>& decoder,
    const std::optional<OptionDict>& decoder_options,
    const int cuda_device_index) {
  AVCodecContextPtr codec_ctx = alloc_codec_context(params->codec_id, decoder);
  configure_codec_context(codec_ctx.get(), params, cuda_device_index);
  codec_ctx->pkt_timebase = pkt_timebase;
  open_codec(codec_ctx.get(), decoder_options);
#ifdef SPDL_USE_CUDA
  if (codec_ctx->hw_device_ctx) {
    codec_ctx->hw_frames_ctx = get_hw_frames_ctx(codec_ctx.get());
  }
#endif
  // TODO: add logging
  return codec_ctx;
}

} // namespace spdl::core::detail
