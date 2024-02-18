#include <libspdl/core/detail/nvdec/decoder.h>

#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/nvdec/utils.h>
#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/logging.h>

#include <folly/logging/xlog.h>

#include <sys/types.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define CLOCKRATE 1

namespace spdl::core::detail {
namespace {
CUvideoctxlock get_lock(CUcontext ctx) {
  CUvideoctxlock lock;
  CHECK_CU(cuvidCtxLockCreate(&lock, ctx), "Failed to create context lock.");
  return lock;
}

CUvideoparserPtr get_parser(
    NvDecDecoder* decoder,
    cudaVideoCodec codec_id,
    unsigned int max_num_decode_surfaces = 1,
    unsigned int max_display_delay = 2,
    bool extract_sei_message = true // temp
) {
  static const auto cb_vseq = [](void* p, CUVIDEOFORMAT* data) -> int {
    return ((NvDecDecoder*)p)->handle_video_sequence(data);
  };
  static const auto cb_decode = [](void* p, CUVIDPICPARAMS* data) -> int {
    return ((NvDecDecoder*)p)->handle_decode_picture(data);
  };
  static const auto cb_disp = [](void* p, CUVIDPARSERDISPINFO* data) -> int {
    return ((NvDecDecoder*)p)->handle_display_picture(data);
  };
  static const auto cb_op = [](void* p, CUVIDOPERATINGPOINTINFO* data) -> int {
    return ((NvDecDecoder*)p)->handle_operating_point(data);
  };
  static const auto cb_sei = [](void* p, CUVIDSEIMESSAGEINFO* data) -> int {
    return ((NvDecDecoder*)p)->handle_sei_msg(data);
  };
  CUVIDPARSERPARAMS parser_params{
      .CodecType = codec_id,
      .ulMaxNumDecodeSurfaces = max_num_decode_surfaces,
      .ulClockRate = CLOCKRATE, // Timestamp units in Hz
      .ulMaxDisplayDelay = max_display_delay,
      .pUserData = (void*)decoder,
      .pfnSequenceCallback = cb_vseq,
      .pfnDecodePicture = cb_decode,
      .pfnDisplayPicture = cb_disp,
      .pfnGetOperatingPoint = cb_op,
      .pfnGetSEIMsg = extract_sei_message ? cb_sei : NULL};
  CUvideoparser parser;
  TRACE_EVENT("nvdec", "cuvidCreateVideoParser");
  CHECK_CU(
      cuvidCreateVideoParser(&parser, &parser_params),
      "Failed to create parser");
  return CUvideoparserPtr{parser};
}

enum RECON { RETAIN, RECONFIGURE, RECREATE };

inline RECON update_type(
    const CUVIDDECODECREATEINFO& i1,
    const CUVIDDECODECREATEINFO& i2) {
  if ( // I/O format or misc decoder config is different
      i1.CodecType != i2.CodecType ||
      i1.DeinterlaceMode != i2.DeinterlaceMode ||
      i1.bitDepthMinus8 != i2.bitDepthMinus8 ||
      i1.ChromaFormat != i2.ChromaFormat ||
      i1.OutputFormat != i2.OutputFormat ||
      i1.ulCreationFlags != i2.ulCreationFlags ||
      i1.ulIntraDecodeOnly != i2.ulIntraDecodeOnly ||
      i1.ulNumOutputSurfaces != i2.ulNumOutputSurfaces ||
      i1.enableHistogram != i2.enableHistogram ||
      // Exceeded the previous maximum width/height
      i1.ulMaxWidth < i2.ulWidth || i1.ulMaxHeight < i2.ulHeight) {
    return RECREATE;
  }
  if (i1.ulWidth == i2.ulWidth && i1.ulHeight == i2.ulHeight &&
      i1.ulTargetWidth == i2.ulTargetWidth &&
      i1.ulTargetHeight == i2.ulTargetHeight &&
      i1.ulNumDecodeSurfaces == i2.ulNumDecodeSurfaces &&
      i1.display_area.left == i2.display_area.left &&
      i1.display_area.top == i2.display_area.top &&
      i1.display_area.right == i2.display_area.right &&
      i1.display_area.bottom == i2.display_area.bottom &&
      i1.target_rect.left == i2.target_rect.left &&
      i1.target_rect.top == i2.target_rect.top &&
      i1.target_rect.right == i2.target_rect.right &&
      i1.target_rect.bottom == i2.target_rect.bottom) {
    return RECON::RETAIN;
  }
  return RECON::RECONFIGURE;
}

inline void warn_if_error(CUvideodecoder decoder, int picture_index) {
  CUVIDGETDECODESTATUS status;
  CUresult result;
  {
    TRACE_EVENT("nvdec", "cuvidGetDecodeStatus");
    result = cuvidGetDecodeStatus(decoder, picture_index, &status);
  }
  if (CUDA_SUCCESS == result) {
    if (status.decodeStatus > cuvidDecodeStatus_Success) {
      XLOG(DBG9) << fmt::format(
          "{} (error code: {})",
          get_desc(status.decodeStatus),
          int(status.decodeStatus));
    }
  }
}
} // namespace

void NvDecDecoder::init(
    CUdevice device_index_,
    cudaVideoCodec codec_,
    CUDABuffer2DPitch* buffer_,
    AVRational timebase_,
    std::tuple<double, double> timestamp_,
    int crop_l,
    int crop_t,
    int crop_r,
    int crop_b,
    int tgt_w,
    int tgt_h) {
  if (crop_l < 0) {
    SPDL_FAIL(fmt::format("crop_left must be non-negative. Found: {}", crop_l));
  }
  if (crop_t < 0) {
    SPDL_FAIL(
        fmt::format("crop_top must be non-negative. Found: {}", crop_top));
  }
  if (crop_r < 0) {
    SPDL_FAIL(
        fmt::format("crop_right must be non-negative. Found: {}", crop_right));
  }
  if (crop_b < 0) {
    SPDL_FAIL(fmt::format(
        "crop_bottom must be non-negative. Found: {}", crop_bottom));
  }
  if (tgt_w > 0 && tgt_w % 2) {
    SPDL_FAIL(fmt::format("target_width must be positive. Found: {}", tgt_w));
  }
  if (tgt_h > 0 && tgt_h % 2) {
    SPDL_FAIL(fmt::format("target_height must be positive. Found: {}", tgt_h));
  }
  if (device != device_index_) {
    device = device_index_;
    cu_ctx = get_cucontext(device);
    lock = get_lock(cu_ctx);
    CHECK_CU(cuCtxSetCurrent(cu_ctx), "Failed to set current context.");

    parser = nullptr;
    decoder = nullptr; // will be re-initialized in the callback
  }
  if (timebase_.num <= 0 || timebase_.den <= 0) {
    SPDL_FAIL_INTERNAL(fmt::format(
        "Invalid time base was found: {}/{}", timebase_.num, timebase_.den));
  }
  if (!parser || codec != codec_) {
    XLOG(DBG9) << "initializing parser";
    codec = codec_;
    parser = get_parser(this, codec);
    decoder = nullptr;
    decoder_param.ulMaxHeight = 0;
    decoder_param.ulMaxWidth = 0;
  }
  buffer = buffer_;
  timebase = timebase_;
  std::tie(start_time, end_time) = timestamp_;

  crop_left = crop_l;
  crop_top = crop_t;
  crop_right = crop_r;
  crop_bottom = crop_b;
  target_width = tgt_w;
  target_height = tgt_h;
}

int NvDecDecoder::handle_video_sequence(CUVIDEOFORMAT* video_fmt) {
  // This function is called by the parser when the first video sequence is
  // received, or when there is a change.
  //
  // The return value of this function is used to update the parser's DPB,
  // (decode picture buffer).
  //  * 0 indicates error,
  //  * 1 indicates no need to update the DPB,
  //  * >1 are the new value for the number of decode surface of the purser.
  //
  // Parser is initialized with a dummy value of
  // min_num_decode_surfaces=1, and, after the first video sequence is
  // processed, the proper minimum number of decoding surfaces are passed to
  // parser via the return value of this function.
  //
  // The argument CUVIDEOFORMAT contains the minimum number of surfaces needed
  // by parser’s DPB (decode picture buffer) for correct decoding.
  //
  // Also, this function initialize/reconfigure/retain/recreate the decoder.
  // The operation to create/destroy/recreate the decoder is very expensive.
  // It is more time-consuming than decoding operations, so we try to
  // reconfigure the decoder whenever it is possible.
  //
  // The decoder can be reconfigured only when the changes are limited to
  // resolutions of input/output sizes (including rescaling and cropping).
  // What is reconfiguable is defined in CUVIDRECONFIGUREDECODERINFO in
  // `cuviddec.h` header file.

  XLOG(DBG9) << print(video_fmt);

  // Check if the input video is supported.
  CUVIDDECODECAPS caps = check_capacity(video_fmt, cap_cache);
  auto output_fmt = get_output_sufrace_format(video_fmt, &caps);

  // TODO: make the following configurable.
  auto max_width = MAX(video_fmt->coded_width, decoder_param.ulMaxWidth);
  auto max_height = MAX(video_fmt->coded_height, decoder_param.ulMaxHeight);

  // Get parameters for creating decoder.
  auto new_decoder_param = get_create_info(
      lock,
      video_fmt,
      output_fmt,
      max_width,
      max_height,
      crop_left,
      crop_top,
      crop_right,
      crop_bottom,
      target_width,
      target_height);

  XLOG(DBG5) << print(&new_decoder_param);

  // Update decoder
  uint ret = [&]() -> uint {
    if (!decoder) {
      decoder.reset(get_decoder(&new_decoder_param));
      return new_decoder_param.ulNumDecodeSurfaces;
    }
    switch (update_type(decoder_param, new_decoder_param)) {
      case RETAIN:
        break;
      case RECONFIGURE:
        reconfigure_decoder(decoder.get(), new_decoder_param);
        break;
      case RECREATE:
        decoder.reset(get_decoder(&new_decoder_param));
        break;
    }
    auto prev_num_surfs = decoder_param.ulNumDecodeSurfaces;
    return prev_num_surfs == new_decoder_param.ulNumDecodeSurfaces
        ? 1
        : new_decoder_param.ulNumDecodeSurfaces;
  }();
  decoder_param = new_decoder_param;

  // Allocate arena
  // TODO: change c and bpp based on codec
  bool channel_last = false;
  size_t c = 1, bpp = 1;
  auto w = decoder_param.ulTargetWidth;
  auto h = decoder_param.ulTargetHeight + decoder_param.ulTargetHeight / 2;
  buffer->allocate(c, h, w, bpp, channel_last);
  return ret;
}

int NvDecDecoder::handle_decode_picture(CUVIDPICPARAMS* pic_params) {
  // This function is called by the parser when the input bit stream is parsed
  // and ready for decodings It just kicks off the decoding work.
  //
  // Return values
  // * 0: fail
  // * >=1: succeess

  // XLOG(INFO) << "Received decoded pictures.";
  // XLOG(INFO) << print(pic_params);
  if (!decoder) {
    SPDL_FAIL_INTERNAL("Decoder not initialized.");
  }
  TRACE_EVENT("nvdec", "cuvidDecodePicture");
  CHECK_CU(
      cuvidDecodePicture(decoder.get(), pic_params),
      "Failed to decode picture.");
  return 1;
}

int NvDecDecoder::handle_display_picture(CUVIDPARSERDISPINFO* disp_info) {
  // This function is called by the parser when the decoding (including
  // post-processing, such as rescaling) is done.
  //
  // The decoded data are still in internal buffer. The `cuvidMapVideoFrame`
  // function makes it accessible via output buffer.
  //
  // The output buffer is a temporary memory region managed by the decoder,
  // so the data must be copied to an application buffer.
  //
  // The output buffer must be then released via `cuvidUnmapVideoFrame`.
  //
  // Return values
  // * 0: fail
  // * >=1: succeess

  // XLOG(INFO) << "Received display pictures.";
  // XLOG(INFO) << print(disp_info);
  double ts = double(disp_info->timestamp) * timebase.num / timebase.den;

  XLOG(DBG9) << fmt::format(
      " --- Frame  PTS={:.3f} ({})", ts, disp_info->timestamp);

  if (ts < start_time || end_time < ts) {
    return 1;
  }

  XLOG(DBG9) << fmt::format(
      "{} x {}", decoder_param.ulTargetWidth, decoder_param.ulTargetHeight);

  warn_if_error(decoder.get(), disp_info->picture_index);
  CUVIDPROCPARAMS proc_params{
      .progressive_frame = disp_info->progressive_frame,
      .second_field = disp_info->repeat_first_field + 1,
      .top_field_first = disp_info->top_field_first,
      .unpaired_field = disp_info->repeat_first_field < 0,
      .output_stream = stream};

  MapGuard mapping(decoder.get(), &proc_params, disp_info->picture_index);

  auto src_frame = (uint8_t*)mapping.frame;
  unsigned int src_pitch = mapping.pitch;
  // Source memory layout
  //
  // <---pitch--->
  // <-width->
  // ┌────────┬───┐  ▲
  // │ YYYYYY │   │ height
  // │ YYYYYY │   │  ▼
  // ├────────┤   │  ▲
  // │ UVUVUV |   │ height / 2
  // └────────┴───┘  ▼
  //

  if (!(decoder_param.bitDepthMinus8 == 8 ||
        decoder_param.ChromaFormat == cudaVideoChromaFormat_420 ||
        decoder_param.OutputFormat == cudaVideoSurfaceFormat_NV12)) {
    SPDL_FAIL(fmt::format(
        "Decoding is not yet implemented for this format. bit_depth={}, chroma_format={}, output_surface_format={}",
        decoder_param.bitDepthMinus8 + 8,
        get_chroma_name(decoder_param.ChromaFormat),
        get_surface_format_name(decoder_param.OutputFormat)));
  }

  // Copy memory
  // Note: arena->H contains both luma and chroma planes.
  CUDA_MEMCPY2D cfg{
      .Height = buffer->h,
      .WidthInBytes = buffer->width_in_bytes,
      .dstArray = nullptr,
      .dstDevice = (CUdeviceptr)buffer->get_next_frame(),
      .dstHost = nullptr,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstPitch = buffer->pitch,
      .dstXInBytes = 0,
      .dstY = 0,
      .srcArray = nullptr,
      .srcDevice = (CUdeviceptr)src_frame,
      .srcHost = nullptr,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcPitch = src_pitch,
      .srcXInBytes = 0,
      .srcY = 0,
  };
  {
    TRACE_EVENT("nvdec", "cuMemcpy2DAsync");
    CHECK_CU(
        cuMemcpy2DAsync(&cfg, stream),
        "Failed to copy Y plane from decoder output surface.");
  }
  buffer->n += 1;

  {
    TRACE_EVENT("nvdec", "cuStreamSynchronize");
    CHECK_CU(cuStreamSynchronize(stream), "Failed to synchronize stream.");
  }
  return 1;
}

int NvDecDecoder::handle_operating_point(CUVIDOPERATINGPOINTINFO* data) {
  // Return values:
  // * <0: fail
  // * >=0: succeess
  //    - bit 0-9: OperatingPoint
  //    - bit 10-10: outputAllLayers
  //    - bit 11-30: reserved

  // XLOG(INFO) << "Received operating points.";

  // Not implemented yet.
  return 0;
}

int NvDecDecoder::handle_sei_msg(CUVIDSEIMESSAGEINFO* msg_info) {
  // Return values:
  // * 0: fail
  // * >=1: succeeded

  // XLOG(INFO) << "Received SEI messages.";
  // XLOG(INFO) << print(msg_info);
  return 0;
}

void NvDecDecoder::decode(
    const uint8_t* data,
    const uint size,
    int64_t pts,
    unsigned long flags) {
  if (!parser) {
    SPDL_FAIL_INTERNAL("Parser is not initialized.");
  }
  CUVIDSOURCEDATAPACKET packet{
      .payload = data, .payload_size = size, .flags = flags, .timestamp = pts};
  TRACE_EVENT("nvdec", "cuvidParseVideoData");
  CHECK_CU(
      cuvidParseVideoData(parser.get(), &packet),
      "Failed to parse video data.");
}

} // namespace spdl::core::detail
