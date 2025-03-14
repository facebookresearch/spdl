/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/nvdec/decoder.h"

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/nvdec/converter.h"
#include "libspdl/core/detail/nvdec/utils.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

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
    NvDecDecoderCore* decoder,
    cudaVideoCodec codec_id,
    unsigned int max_num_decode_surfaces = 1,
    unsigned int max_display_delay = 2,
    bool extract_sei_message = true // temp
) {
  static const auto cb_vseq = [](void* p, CUVIDEOFORMAT* data) -> int {
    return ((NvDecDecoderCore*)p)->handle_video_sequence(data);
  };
  static const auto cb_decode = [](void* p, CUVIDPICPARAMS* data) -> int {
    return ((NvDecDecoderCore*)p)->handle_decode_picture(data);
  };
  static const auto cb_disp = [](void* p, CUVIDPARSERDISPINFO* data) -> int {
    return ((NvDecDecoderCore*)p)->handle_display_picture(data);
  };
  static const auto cb_op = [](void* p, CUVIDOPERATINGPOINTINFO* data) -> int {
    return ((NvDecDecoderCore*)p)->handle_operating_point(data);
  };
  static const auto cb_sei = [](void* p, CUVIDSEIMESSAGEINFO* data) -> int {
    return ((NvDecDecoderCore*)p)->handle_sei_msg(data);
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
      .pfnGetSEIMsg = extract_sei_message
          ? cb_sei
          : static_cast<PFNVIDSEIMSGCALLBACK>(nullptr),
  };
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
    // VLOG(9) << "Recreating the decoder object.\n    "
    //            << detail::get_diff(i1, i2);
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
  // VLOG(9) << "Reconfiguring the decoder object.";
  return RECON::RECONFIGURE;
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

inline void warn_if_error(CUvideodecoder decoder, int picture_index) {
  CUVIDGETDECODESTATUS status;
  CUresult result;
  {
    TRACE_EVENT("nvdec", "cuvidGetDecodeStatus");
    result = cuvidGetDecodeStatus(decoder, picture_index, &status);
  }
  if (CUDA_SUCCESS == result) {
    if (status.decodeStatus > cuvidDecodeStatus_Success) {
      VLOG(9) << fmt::format(
          "{} (error code: {})",
          get_desc(status.decodeStatus),
          int(status.decodeStatus));
    }
  }
}

std::tuple<double, double> NO_WINDOW{
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};
} // namespace

////////////////////////////////////////////////////////////////////////////////
// NvDecDecoderCore
////////////////////////////////////////////////////////////////////////////////

void NvDecDecoderCore::init(
    CUdevice device_index_,
    cudaVideoCodec codec_,
    Rational timebase_,
    const std::optional<std::tuple<double, double>>& timestamp_,
    CropArea crop_,
    int tgt_w,
    int tgt_h,
    const std::optional<std::string>& pix_fmt_) {
  if (crop_.left < 0) {
    SPDL_FAIL(
        fmt::format("crop_left must be non-negative. Found: {}", crop_.left));
  }
  if (crop_.top < 0) {
    SPDL_FAIL(
        fmt::format("crop_top must be non-negative. Found: {}", crop_.top));
  }
  if (crop_.right < 0) {
    SPDL_FAIL(
        fmt::format("crop_right must be non-negative. Found: {}", crop_.right));
  }
  if (crop_.bottom < 0) {
    SPDL_FAIL(fmt::format(
        "crop_bottom must be non-negative. Found: {}", crop_.bottom));
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
    VLOG(9) << "initializing parser";
    codec = codec_;
    parser = get_parser(this, codec);
    decoder = nullptr;
    decoder_param.ulMaxHeight = 720;
    decoder_param.ulMaxWidth = 1280;
  }
  timebase = timebase_;
  std::tie(start_time, end_time) = timestamp_ ? *timestamp_ : NO_WINDOW;

  crop = crop_;
  target_width = tgt_w;
  target_height = tgt_h;
  pix_fmt = pix_fmt_;
}

int NvDecDecoderCore::handle_video_sequence(CUVIDEOFORMAT* video_fmt) {
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
  if (cb_disabled) {
    return 1;
  }
  TRACE_EVENT("nvdec", "handle_video_sequence");

  VLOG(9) << print(video_fmt);

  // Check if the input video is supported.
  CUVIDDECODECAPS caps = check_capacity(video_fmt, cap_cache);
  auto output_fmt = get_output_sufrace_format(video_fmt, &caps);

  auto max_width = MAX(video_fmt->coded_width, decoder_param.ulMaxWidth);
  auto max_height = MAX(video_fmt->coded_height, decoder_param.ulMaxHeight);

  // Get parameters for creating decoder.
  auto new_decoder_param = get_create_info(
      lock,
      video_fmt,
      output_fmt,
      max_width,
      max_height,
      crop,
      target_width,
      target_height);

  VLOG(5) << print(&new_decoder_param);

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

  converter = get_converter(
      stream,
      tracker,
      &decoder_param,
      video_fmt->video_signal_description.matrix_coefficients,
      pix_fmt);
  return ret;
}

int NvDecDecoderCore::handle_decode_picture(CUVIDPICPARAMS* pic_params) {
  // This function is called by the parser when the input bit stream is parsed
  // and ready for decodings It just kicks off the decoding work.
  //
  // Return values
  // * 0: fail
  // * >=1: succeess

  if (cb_disabled) {
    return 1;
  }
  TRACE_EVENT("nvdec", "handle_decode_picture");

  // LOG(INFO) << "Received decoded pictures.";
  // LOG(INFO) << print(pic_params);
  if (!decoder) {
    SPDL_FAIL_INTERNAL("Decoder not initialized.");
  }
  TRACE_EVENT("nvdec", "cuvidDecodePicture");
  CHECK_CU(
      cuvidDecodePicture(decoder.get(), pic_params),
      "Failed to decode picture.");
  return 1;
}

int NvDecDecoderCore::handle_display_picture(CUVIDPARSERDISPINFO* disp_info) {
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

  if (cb_disabled) {
    return 1;
  }
  TRACE_EVENT("nvdec", "handle_display_picture");

  // LOG(INFO) << "Received display pictures.";
  // LOG(INFO) << print(disp_info);
  double ts = double(disp_info->timestamp) * timebase.num / timebase.den;

  VLOG(9) << fmt::format(
      " --- Frame  PTS={:.3f} ({})", ts, disp_info->timestamp);

  if (ts < start_time || end_time <= ts) {
    return 1;
  }

  VLOG(9) << fmt::format(
      "{} x {}", decoder_param.ulTargetWidth, decoder_param.ulTargetHeight);

  warn_if_error(decoder.get(), disp_info->picture_index);
  CUVIDPROCPARAMS proc_params{
      .progressive_frame = disp_info->progressive_frame,
      .second_field = disp_info->repeat_first_field + 1,
      .top_field_first = disp_info->top_field_first,
      .unpaired_field = disp_info->repeat_first_field < 0,
      .output_stream = stream};

  // Make the decoded frame available to output surface
  MapGuard mapping(decoder.get(), &proc_params, disp_info->picture_index);

  // Copy the surface to user-owning buffer
  converter->convert((uint8_t*)mapping.frame, mapping.pitch);
  tracker->i += 1;

  TRACE_EVENT("nvdec", "cuStreamSynchronize");
  CHECK_CU(cuStreamSynchronize(stream), "Failed to synchronize stream.");
  return 1;
}

int NvDecDecoderCore::handle_operating_point(CUVIDOPERATINGPOINTINFO* data) {
  // Return values:
  // * <0: fail
  // * >=0: succeess
  //    - bit 0-9: OperatingPoint
  //    - bit 10-10: outputAllLayers
  //    - bit 11-30: reserved

  if (cb_disabled) {
    return 1;
  }
  TRACE_EVENT("nvdec", "handle_operating_point");

  // LOG(INFO) << "Received operating points.";

  // Not implemented yet.
  return 0;
}

int NvDecDecoderCore::handle_sei_msg(CUVIDSEIMESSAGEINFO* msg_info) {
  // Return values:
  // * 0: fail
  // * >=1: succeeded

  if (cb_disabled) {
    return 1;
  }

  // LOG(INFO) << "Received SEI messages.";
  // LOG(INFO) << print(msg_info);
  return 0;
}

void NvDecDecoderCore::decode(
    const uint8_t* data,
    const uint size,
    int64_t pts,
    unsigned long flags) {
  if (!parser) {
    SPDL_FAIL_INTERNAL("Parser is not initialized.");
  }
  CUVIDSOURCEDATAPACKET packet{
      .flags = flags, .payload_size = size, .payload = data, .timestamp = pts};
  TRACE_EVENT("nvdec", "cuvidParseVideoData");
  CHECK_CU(
      cuvidParseVideoData(parser.get(), &packet),
      "Failed to parse video data.");
}

void NvDecDecoderCore::reset() {
  if (!parser) {
    return;
  }

  cb_disabled = true;

  const unsigned char data{};
  CUVIDSOURCEDATAPACKET packet{
      .flags = CUVID_PKT_ENDOFSTREAM,
      .payload_size = 0,
      .payload = &data,
      .timestamp = 0};
  TRACE_EVENT("nvdec", "cuvidParseVideoData");
  CHECK_CU(
      cuvidParseVideoData(parser.get(), &packet),
      "Failed to parse video data.");

  cb_disabled = false;
}

////////////////////////////////////////////////////////////////////////////////
// NvDecDecoderInternal
////////////////////////////////////////////////////////////////////////////////

void NvDecDecoderInternal::reset() {
  core.reset();
}

void NvDecDecoderInternal::init(
    int device_index,
    enum AVCodecID codec_id,
    Rational time_base,
    const std::optional<std::tuple<double, double>>& timestamp,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt) {
  core.init(
      device_index,
      covert_codec_id(codec_id),
      time_base,
      timestamp,
      crop,
      target_width,
      target_height,
      pix_fmt);
}

// TEMP
CUDABufferTracker get_buffer_tracker(
    const CUDAConfig& cuda_config,
    size_t num_packets,
    AVCodecParameters* codecpar,
    const CropArea& crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt,
    bool is_image);

template <MediaType media_type>
void NvDecDecoderInternal::decode_packets(
    PacketsPtr<media_type> packets,
    CUDABufferTracker& tracker) {
  TRACE_EVENT("nvdec", "decode_packets");

  auto num_packets = packets->num_packets();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  core.tracker = &tracker;

  size_t it = 0;
  unsigned long flags = CUVID_PKT_TIMESTAMP;
  switch (packets->codecpar->codec_id) {
    case AV_CODEC_ID_MPEG4: {
      // TODO: Add special handling par
      // Video_Codec_SDK_12.1.14/blob/main/Samples/Utils/FFmpegDemuxer.h#L326-L345
      // TODO: Test this with MP4 file.
      SPDL_FAIL("NOT IMPLEMENTED.");
      ++it;
    }
    case AV_CODEC_ID_AV1:
      // TODO handle
      // https://github.com/FFmpeg/FFmpeg/blob/5e2b0862eb1d408625232b37b7a2420403cd498f/libavcodec/cuviddec.c#L1001-L1009
      SPDL_FAIL("NOT IMPLEMENTED.");
      ++it;
    default:;
  }

#define _PKT(i) packets->get_packets()[i]
#define _PTS(pkt)                                           \
  (static_cast<double>(pkt->pts) * packets->time_base.num / \
   packets->time_base.den)

  flags |= CUVID_PKT_ENDOFPICTURE;
  for (; it < num_packets - 1; ++it) {
    auto pkt = _PKT(it);
    VLOG(9) << fmt::format(" -- packet  PTS={:.3f} ({})", _PTS(pkt), pkt->pts);

    core.decode(pkt->data, pkt->size, pkt->pts, flags);
  }
  auto pkt = _PKT(it);
  flags |= CUVID_PKT_ENDOFSTREAM;
  core.decode(pkt->data, pkt->size, pkt->pts, flags);

#undef _PKT
#undef _PTS

  VLOG(5) << fmt::format(
      "Decoded {} frames from {} packets.", tracker.i, packets->num_packets());

  if constexpr (media_type == MediaType::Video) {
    // We preallocated the buffer with the number of packets, but these packets
    // could contains the frames outside of specified timestamps.
    // So we update the shape with the actual number of frames.
    tracker.buffer->shape[0] = tracker.i;
  }
}

template <MediaType media_type>
CUDABufferPtr NvDecDecoderInternal::decode(
    PacketsPtr<media_type> packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt) {
  TRACE_EVENT("nvdec", "decode");

  ensure_cuda_initialized();

  auto num_packets = packets->num_packets();

  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  auto tracker = detail::get_buffer_tracker(
      cuda_config,
      num_packets,
      packets->codecpar,
      crop,
      target_width,
      target_height,
      pix_fmt,
      media_type == MediaType::Image);

  decode_packets(std::move(packets), tracker);

  return std::move(tracker.buffer);
}

template CUDABufferPtr NvDecDecoderInternal::decode(
    PacketsPtr<MediaType::Video> packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt);

template CUDABufferPtr NvDecDecoderInternal::decode(
    PacketsPtr<MediaType::Image> packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt);

} // namespace spdl::core::detail
