#include <libspdl/core/detail/nvdec/utils.h>

#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/logging.h>

#include <fmt/format.h>
#include <folly/logging/xlog.h>

#include <span>

namespace spdl::core::detail {

cudaVideoCodec covert_codec_id(AVCodecID id) {
  switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO:
      return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO:
      return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4:
      return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3:
    case AV_CODEC_ID_VC1:
      return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264:
      return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC:
      return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8:
      return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9:
      return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG:
      return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1:
      return cudaVideoCodec_AV1;
    default:
      return cudaVideoCodec_NumCodecs;
  }
}

bool is_compatible(const CUVIDEOFORMAT* fmt, const CUVIDDECODECAPS& caps) {
  return caps.eCodecType == fmt->codec &&
      caps.eChromaFormat == fmt->chroma_format &&
      caps.nBitDepthMinus8 == fmt->bit_depth_luma_minus8;
}

void check_support(CUVIDEOFORMAT* fmt, CUVIDDECODECAPS caps) {
  if (!caps.bIsSupported) {
    SPDL_FAIL(fmt::format(
        "Codec not supported on this GPU. Codec: {}, Bit Depth: {}, Chroma Format: {}",
        get_codec_name(fmt->codec),
        fmt->bit_depth_luma_minus8 + 8,
        get_chroma_name(fmt->chroma_format)));
  }
  if ((fmt->coded_width < caps.nMinWidth) ||
      (fmt->coded_width > caps.nMaxWidth) ||
      (fmt->coded_height < caps.nMinHeight) ||
      (fmt->coded_height > caps.nMaxHeight)) {
    SPDL_FAIL(fmt::format(
        "Resolution is outside of the supported range for this GPU. "
        "Input video resolution is {}x{} (wxh). "
        "The minimum/maximum supported resolutions are {}x{}, {}x{}",
        fmt->coded_width,
        fmt->coded_height,
        caps.nMinWidth,
        caps.nMinHeight,
        caps.nMaxWidth,
        caps.nMaxHeight));
  }
  if (auto mb_count = (fmt->coded_width >> 4) * (fmt->coded_height >> 4);
      mb_count > caps.nMaxMBCount) {
    SPDL_FAIL(fmt::format(
        "Number of macroblocks too large for this GPU. "
        "Input video macroblock count {}. "
        "Maximum supported number of macroblocks {}.",
        mb_count,
        caps.nMaxMBCount));
  }
}

void reconfigure_decoder(
    CUvideodecoder decoder,
    const CUVIDDECODECREATEINFO& create_info) {
  CUVIDRECONFIGUREDECODERINFO reconf = {
      .ulWidth = (unsigned int)create_info.ulWidth,
      .ulHeight = (unsigned int)create_info.ulHeight,
      .ulTargetWidth = (unsigned int)create_info.ulTargetWidth,
      .ulTargetHeight = (unsigned int)create_info.ulTargetHeight,
      .ulNumDecodeSurfaces = (unsigned int)create_info.ulNumDecodeSurfaces,
      .display_area =
          {
              .left = create_info.display_area.left,
              .top = create_info.display_area.top,
              .right = create_info.display_area.right,
              .bottom = create_info.display_area.bottom,
          },
      .target_rect = {
          .left = create_info.target_rect.left,
          .top = create_info.target_rect.top,
          .right = create_info.target_rect.right,
          .bottom = create_info.target_rect.bottom,
      }};
  TRACE_EVENT("nvdec", "cuvidReconfigureDecoder");
  CHECK_CU(
      cuvidReconfigureDecoder(decoder, &reconf),
      "Failed to reconfigure decoder.");
}

CUvideodecoder get_decoder(CUVIDDECODECREATEINFO* param) {
  CUvideodecoder decoder;
  TRACE_EVENT("nvdec", "cuvidCreateDecoder");
  CHECK_CU(cuvidCreateDecoder(&decoder, param), "Failed to create decoder.");
  XLOG(DBG9) << "Created CUvideodecoder: " << decoder;
  return decoder;
}

CUVIDDECODECREATEINFO get_create_info(
    CUvideoctxlock lock,
    CUVIDEOFORMAT* video_fmt,
    cudaVideoSurfaceFormat output_fmt,
    unsigned int max_width,
    unsigned int max_height,
    int crop_left,
    int crop_top,
    int crop_right,
    int crop_bottom,
    int target_width,
    int target_height) {
  // XLOG(INFO) << fmt::format(
  //    "Output sufrace format: {}", get_surface_format_name(output_fmt));

  int width = video_fmt->display_area.right - video_fmt->display_area.left -
      crop_left - crop_right;
  int height = video_fmt->display_area.bottom - video_fmt->display_area.top -
      crop_top - crop_bottom;
  if (width <= 0) {
    SPDL_FAIL(fmt::format(
        "Invalid image width: {} (source width: {}, crop_left: {}, crop_right: {})",
        width,
        video_fmt->display_area.right - video_fmt->display_area.left,
        crop_left,
        crop_right));
  }
  if (height <= 0) {
    SPDL_FAIL(fmt::format(
        "Invalid image height: {} (source height: {}, crop_top: {}, crop_bottom: {})",
        height,
        video_fmt->display_area.bottom - video_fmt->display_area.top,
        crop_top,
        crop_bottom));
  }

  // Note: The frame is first cropped then resized to target_width/height

  uint tgt_w = target_width > 0 ? target_width : width;
  uint tgt_h = target_height > 0 ? target_height : height;
  // make evan
  // target_width and target_height are already checked to be even.
  // This is for case when the native size are not even.
  /*
  if (output_fmt == cudaVideoSurfaceFormat_NV12 ||
      output_fmt == cudaVideoSurfaceFormat_P016) {
    if (tgt_w % 2) {
      tgt_w -= 1;
      XLOG_FIRST_N(WARN, 1)
          << fmt::format("Width must be even. Cropping to {}.", tgt_w);
    }
    if (tgt_h % 2) {
      tgt_h -= 1;
      XLOG_FIRST_N(WARN, 1)
          << fmt::format("Height must be even. Cropping to {}.", tgt_h);
    }
  }
  */
  return CUVIDDECODECREATEINFO{
      .ulWidth = video_fmt->coded_width,
      .ulHeight = video_fmt->coded_height,
      .ulNumDecodeSurfaces = video_fmt->min_num_decode_surfaces,
      .CodecType = video_fmt->codec,
      .ChromaFormat = video_fmt->chroma_format,
      .ulCreationFlags = cudaVideoCreate_PreferCUVID, // TODO: Check other flags
      .bitDepthMinus8 = video_fmt->bit_depth_luma_minus8,
      .ulIntraDecodeOnly = 0,
      .ulMaxWidth = max_width,
      .ulMaxHeight = max_height,
      .display_area =
          {.left = (short)(video_fmt->display_area.left + crop_left),
           .top = (short)(video_fmt->display_area.top + crop_top),
           .right = (short)(video_fmt->display_area.right - crop_right),
           .bottom = (short)(video_fmt->display_area.bottom - crop_bottom)},
      .OutputFormat = output_fmt,
      .DeinterlaceMode = video_fmt->progressive_sequence
          ? cudaVideoDeinterlaceMode_Weave
          : cudaVideoDeinterlaceMode_Adaptive,
      .ulTargetWidth = tgt_w,
      .ulTargetHeight = tgt_h,
      .ulNumOutputSurfaces = 2,
      .vidLock = lock,
      // note:
      // lock is required when using cudaVideoCreate_PreferCUDA creation flag
      .target_rect =
          {.left = 0, .top = 0, .right = (short)tgt_w, .bottom = (short)tgt_h}
      //  The aspect ratio is 1:1 to rescaled frame.
  };
}

cudaVideoSurfaceFormat get_output_sufrace_format(
    CUVIDEOFORMAT* video_fmt,
    CUVIDDECODECAPS* decode_caps) {
  cudaVideoSurfaceFormat surface_format = [&]() {
    switch (video_fmt->chroma_format) {
      case cudaVideoChromaFormat_Monochrome:
      case cudaVideoChromaFormat_420: {
        return video_fmt->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016
                                                : cudaVideoSurfaceFormat_NV12;
      }
      case cudaVideoChromaFormat_444:
        return video_fmt->bit_depth_luma_minus8
            ? cudaVideoSurfaceFormat_YUV444_16Bit
            : cudaVideoSurfaceFormat_YUV444;
      case cudaVideoChromaFormat_422:
        // YUV422 as output is not supported, so we use NV12
        return cudaVideoSurfaceFormat_NV12;
    }
  }();

  // Check if output format supported.
  if (decode_caps->nOutputFormatMask & (1 << surface_format)) {
    return surface_format;
  }

  // Pick a fallback option.
  if (decode_caps->nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12)) {
    return cudaVideoSurfaceFormat_NV12;
  }
  if (decode_caps->nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016)) {
    return cudaVideoSurfaceFormat_P016;
  }
  if (decode_caps->nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444)) {
    return cudaVideoSurfaceFormat_YUV444;
  }
  if (decode_caps->nOutputFormatMask &
      (1 << cudaVideoSurfaceFormat_YUV444_16Bit)) {
    return cudaVideoSurfaceFormat_YUV444_16Bit;
  }
  SPDL_FAIL("No supported output format found.");
}

// clang-format off
// Appendix: Cpacities
//
// A100:
// Codec  JPEG   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  32768  MaxHeight  16384  MaxMBCount  67108864  MinWidth  64   MinHeight  64   SurfaceFormat  NV12
// Codec  MPEG1  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4080   MaxHeight  4080   MaxMBCount  65280     MinWidth  48   MinHeight  16   SurfaceFormat  NV12
// Codec  MPEG2  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4080   MaxHeight  4080   MaxMBCount  65280     MinWidth  48   MinHeight  16   SurfaceFormat  NV12
// Codec  MPEG4  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  2032   MaxHeight  2032   MaxMBCount  8192      MinWidth  48   MinHeight  16   SurfaceFormat  NV12
// Codec  H264   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4096   MaxHeight  4096   MaxMBCount  65536     MinWidth  48   MinHeight  16   SurfaceFormat  NV12
// Codec  HEVC   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12
// Codec  HEVC   BitDepth  10  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12 P016
// Codec  HEVC   BitDepth  12  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12 P016
// Codec  HEVC   BitDepth  8   ChromaFormat  4:4:4  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  YUV444
// Codec  HEVC   BitDepth  10  ChromaFormat  4:4:4  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  YUV444P16
// Codec  HEVC   BitDepth  12  ChromaFormat  4:4:4  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  YUV444P16
// Codec  VC1    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  2032   MaxHeight  2032   MaxMBCount  8192      MinWidth  48   MinHeight  16   SurfaceFormat  NV12
// Codec  VP8    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4096   MaxHeight  4096   MaxMBCount  65536     MinWidth  48   MinHeight  16   SurfaceFormat  NV12
// Codec  VP9    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12
// Codec  VP9    BitDepth  10  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12 P016
// Codec  VP9    BitDepth  12  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12 P016
// Codec  AV1    BitDepth  8   ChromaFormat  4:2:0  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A
// Codec  AV1    BitDepth  10  ChromaFormat  4:2:0  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A
// Codec  AV1    BitDepth  8   ChromaFormat  4:0:0  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A
// Codec  AV1    BitDepth  10  ChromaFormat  4:0:0  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A
// clang-format on
CUVIDDECODECAPS check_capacity(
    CUVIDEOFORMAT* video_fmt,
    std::vector<CUVIDDECODECAPS>& cache) {
  // Check cache first
  for (auto& caps : cache) {
    if (is_compatible(video_fmt, caps)) {
      check_support(video_fmt, caps);
      return caps;
    }
  }
  CUVIDDECODECAPS caps{
      .eCodecType = video_fmt->codec,
      .eChromaFormat = video_fmt->chroma_format,
      .nBitDepthMinus8 = video_fmt->bit_depth_luma_minus8};
  {
    TRACE_EVENT("nvdec", "cuvidGetDecoderCaps");
    CHECK_CU(cuvidGetDecoderCaps(&caps), "Failed to get decoder capabilities.");
  }
  check_support(video_fmt, caps);
  cache.push_back(caps);
  return caps;
}

const char* get_desc(cuvidDecodeStatus status) {
  switch (status) {
    case cuvidDecodeStatus_Invalid:
      return "Decode status is not valid.";
    case cuvidDecodeStatus_InProgress:
      return "Decode is in progress.";
    case cuvidDecodeStatus_Success:
      return "Decode is completed without an error.";
    case cuvidDecodeStatus_Error:
      return "Decode is completed with an unconcealed error.";
    case cuvidDecodeStatus_Error_Concealed:
      return "Decode is completed with a concealed error.";
    default:
      return "Unkonwn decode status.";
  }
}

const char* get_codec_name(cudaVideoCodec codec) {
  switch (codec) {
    case cudaVideoCodec_MPEG1:
      return "MPEG1";
    case cudaVideoCodec_MPEG2:
      return "MPEG2";
    case cudaVideoCodec_MPEG4:
      return "MPEG4";
    case cudaVideoCodec_VC1:
      return "VC1";
    case cudaVideoCodec_H264:
      return "H264";
    case cudaVideoCodec_JPEG:
      return "JPEG";
    case cudaVideoCodec_H264_SVC:
      return "H264_SVC";
    case cudaVideoCodec_H264_MVC:
      return "H264_MVC";
    case cudaVideoCodec_HEVC:
      return "HEVC";
    case cudaVideoCodec_VP8:
      return "VP8";
    case cudaVideoCodec_VP9:
      return "VP9";
    case cudaVideoCodec_AV1:
      return "AV1";
    case cudaVideoCodec_NumCodecs:
      return "NumCodecs";
    case cudaVideoCodec_YUV420:
      return "YUV420";
    case cudaVideoCodec_YV12:
      return "YV12";
    case cudaVideoCodec_NV12:
      return "NV12";
    case cudaVideoCodec_YUYV:
      return "YUYV";
    case cudaVideoCodec_UYVY:
      return "UYVY";
  }
}

const char* get_chroma_name(cudaVideoChromaFormat chroma) {
  switch (chroma) {
    case cudaVideoChromaFormat_Monochrome:
      return "Monochrome";
    case cudaVideoChromaFormat_420:
      return "420";
    case cudaVideoChromaFormat_422:
      return "422";
    case cudaVideoChromaFormat_444:
      return "444";
  }
}

const char* get_video_signal_format_name(unsigned char video_format) {
  std::vector<const char*> names{
      "Component", "PAL", "NTSC", "SECAM", "MAC", "Unspecified"};
  if (video_format >= sizeof(names)) {
    return "Unknown";
  }
  return names[video_format];
}

std::string print(const CUVIDEOFORMAT* video_fmt) {
  return fmt::format(
      R"EOF(CUVIDEOFORMAT(
  .codec = {}
  .frame_rate = {} / {}
  .progressive_sequence = {} ({})
  .bit_depth_luma_minus8 = {}
  .bit_depth_chroma_minus8 = {}
  .min_num_decode_surfaces = {}
  .coded_width = {}
  .coded_height = {}
  .display_area = (
    .left = {}
    .top = {}
    .right = {}
    .bottom = {}
  )
  .chroma_format = {}
  .bitrate = {}
  .display_aspect_ratio = (
    .x = {}
    .y = {}
  )
  .video_signal_description = (
    .video_format = {} ({})
    .video_full_range_flag = {}
    .reserved_zero_bits = {}
    .color_primaries = {}
    .transfer_characteristics = {}
    .matrix_coefficients = {}
  )
  .seqhdr_data_length = {}
))EOF",
      get_codec_name(video_fmt->codec),
      video_fmt->frame_rate.numerator,
      video_fmt->frame_rate.denominator,
      video_fmt->progressive_sequence,
      video_fmt->progressive_sequence ? "progressive" : "interlaced",
      video_fmt->bit_depth_luma_minus8,
      video_fmt->bit_depth_chroma_minus8,
      video_fmt->min_num_decode_surfaces,
      video_fmt->coded_width,
      video_fmt->coded_height,
      video_fmt->display_area.left,
      video_fmt->display_area.top,
      video_fmt->display_area.right,
      video_fmt->display_area.bottom,
      get_chroma_name(video_fmt->chroma_format),
      video_fmt->bitrate,
      video_fmt->display_aspect_ratio.x,
      video_fmt->display_aspect_ratio.y,
      (unsigned char)video_fmt->video_signal_description.video_format,
      get_video_signal_format_name(
          video_fmt->video_signal_description.video_format),
      (unsigned char)video_fmt->video_signal_description.video_full_range_flag,
      (unsigned char)video_fmt->video_signal_description.reserved_zero_bits,
      video_fmt->video_signal_description.color_primaries,
      video_fmt->video_signal_description.transfer_characteristics,
      video_fmt->video_signal_description.matrix_coefficients,
      video_fmt->seqhdr_data_length);
}

const char* get_surface_format_name(cudaVideoSurfaceFormat s) {
  switch (s) {
    case cudaVideoSurfaceFormat_NV12:
      return "NV12";
    case cudaVideoSurfaceFormat_P016:
      return "P016";
    case cudaVideoSurfaceFormat_YUV444:
      return "YUV444";
    case cudaVideoSurfaceFormat_YUV444_16Bit:
      return "YUV444_16Bit";
  }
}

std::string print(const CUVIDDECODECAPS* decode_caps) {
  return fmt::format(
      R"EOF(CUVIDDECODECAPS(
  .eCodecType = {}
  .eChromaFormat = {}
  .nBitDepthMinus8 = {}
  .bIsSupported = {}
  .nNumNVDECs = {}
  .nOutputFormatMask = {:#018b} ({})
  .nMaxWidth = {}
  .nMaxHeight = {}
  .nMaxMBCount = {}
  .nMinWidth = {}
  .nMinHeight = {}
  .bIsHistogramSupported = {}
  .nCounterBitDepth = {}
  .nMaxHistogramBins = {}
))EOF",
      get_codec_name(decode_caps->eCodecType),
      get_chroma_name(decode_caps->eChromaFormat),
      decode_caps->nBitDepthMinus8,
      decode_caps->bIsSupported,
      decode_caps->nNumNVDECs,
      decode_caps->nOutputFormatMask,
      get_surface_format_name(
          (cudaVideoSurfaceFormat)(decode_caps->nOutputFormatMask >> 1)),
      decode_caps->nMaxWidth,
      decode_caps->nMaxHeight,
      decode_caps->nMaxMBCount,
      decode_caps->nMinWidth,
      decode_caps->nMinHeight,
      decode_caps->bIsHistogramSupported,
      decode_caps->nCounterBitDepth,
      decode_caps->nMaxHistogramBins);
}

std::string print(const CUVIDPICPARAMS* disp_info) {
  return fmt::format(
      R"EOF(CUVIDPARSERDISPINFO(
  .PicWidthInMbs = {}
  .FrameHeightInMbs = {}
  .CurrPicIdx = {}
  .field_pic_flag = {}
  .bottom_field_flag = {}
  .second_field = {}
  .nBitstreamDataLen = {}
  .pBitstreamData = ...
  .nNumSlices = {}
  .pSliceDataOffsets = {}
  .ref_pic_flag = {}
  .intra_pic_flag = {}
))EOF",
      disp_info->PicWidthInMbs,
      disp_info->FrameHeightInMbs,
      disp_info->CurrPicIdx,
      disp_info->field_pic_flag,
      disp_info->bottom_field_flag,
      disp_info->second_field,
      disp_info->nBitstreamDataLen,
      // disp_info->pBitstreamData,
      disp_info->nNumSlices,
      fmt::join(
          std::span(
              disp_info->pSliceDataOffsets,
              disp_info->pSliceDataOffsets + disp_info->nNumSlices),
          ", "),
      disp_info->ref_pic_flag,
      disp_info->intra_pic_flag);
}
namespace {
std::string print_repeat_first_field(int i) {
  if (i == 1) {
    return "ivtc";
  }
  if (i == 2) {
    return "frame doubling";
  }
  if (i == 4) {
    return "frame tripling";
  }
  return "unpaired field";
}
} // namespace

std::string print(const CUVIDPARSERDISPINFO* parser_disp) {
  return fmt::format(
      R"EOF(CUVIDPARSERDISPINFO(
  .picture_index = {}
  .progressive_frame = {}
  .top_field_first = {}
  .repeat_first_field = {} ({})
  .timestamp = {}
))EOF",
      parser_disp->picture_index,
      parser_disp->progressive_frame,
      parser_disp->top_field_first,
      parser_disp->repeat_first_field,
      print_repeat_first_field(parser_disp->repeat_first_field),
      parser_disp->timestamp);
}

std::string print(const CUVIDSEIMESSAGEINFO* msg_info) {
  return fmt::format(
      R"EOF(CUVIDSEIMESSAGEINFO(
  .pSEIData = {}
  .pSEIMessage = (
    .sei_message_type = {}
    .sei_message_size = {}
  )
  .sei_message_count = {}
  .picIdx = {}
))EOF",
      msg_info->pSEIData,
      msg_info->pSEIMessage->sei_message_type,
      msg_info->pSEIMessage->sei_message_size,
      msg_info->sei_message_count,
      msg_info->picIdx);
}

std::string print(const CUVIDDECODECREATEINFO* p) {
  return fmt::format(
      R"EOF(CUVIDDECODECREATEINFO(
  .ulWidth = {}
  .ulHeight = {}
  .ulNumDecodeSurfaces = {}
  .CodecType = {}
  .ChromaFormat = {}
  .ulCreationFlags = {}
  .ulMaxWidth = {}
  .ulMaxHeight = {}
  .display_area = (
    .left = {}
    .top = {}
    .right = {}
    .bottom = {}
  )
  .OutputFormat = {}
  .DeinterlaceMode = {}
  .ulTargetWidth = {}
  .ulTargetHeight = {}
  .ulNumOutputSurfaces = {}
  .target_rect = (
    .left = {}
    .top = {}
    .right = {}
    .bottom = {}
  )
  .enableHistogram = {}
))EOF",
      p->ulWidth,
      p->ulHeight,
      p->ulNumDecodeSurfaces,
      get_codec_name(p->CodecType),
      get_chroma_name(p->ChromaFormat),
      p->ulCreationFlags,
      p->ulMaxWidth,
      p->ulMaxHeight,
      p->display_area.left,
      p->display_area.top,
      p->display_area.right,
      p->display_area.bottom,
      get_surface_format_name(p->OutputFormat),
      p->DeinterlaceMode,
      p->ulTargetWidth,
      p->ulTargetHeight,
      p->ulNumOutputSurfaces,
      p->target_rect.left,
      p->target_rect.top,
      p->target_rect.right,
      p->target_rect.bottom,
      p->enableHistogram);
}

} // namespace spdl::core::detail
