/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/codec.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include "libspdl/cuda/detail/utils.h"
#include "libspdl/cuda/nvdec/detail/decoder.h"
#include "libspdl/cuda/nvdec/detail/utils.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <sys/types.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define CLOCKRATE 1

namespace spdl::cuda::detail {
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
} // namespace

////////////////////////////////////////////////////////////////////////////////
// NvDecDecoderCore
////////////////////////////////////////////////////////////////////////////////

void NvDecDecoderCore::init(
    const CUDAConfig& device_config_,
    const spdl::core::VideoCodec& codec_,
    const CropArea& crop_,
    int tgt_w,
    int tgt_h) {
  if (auto tb = codec_.get_time_base(); tb.num <= 0 || tb.den <= 0) {
    SPDL_FAIL_INTERNAL(
        fmt::format("Invalid time base was found: {}/{}", tb.num, tb.den));
  }
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
  if (device_config.device_index != device_config_.device_index) {
    device_config = device_config_;
    cu_ctx = get_cucontext(device_config.device_index);
    lock = get_lock(cu_ctx);
    CHECK_CU(cuCtxSetCurrent(cu_ctx), "Failed to set current context.");

    parser = nullptr;
    decoder = nullptr; // will be re-initialized in the callback
  }

  auto cdc = convert_codec_id(codec_.get_codec_id());
  if (!parser || codec != cdc) {
    VLOG(9) << "initializing parser";
    codec = cdc;
    parser = get_parser(this, codec);
    decoder = nullptr;
    decoder_param.ulMaxHeight = 720;
    decoder_param.ulMaxWidth = 1280;
  }

  src_width = codec_.get_width();
  src_height = codec_.get_height();
  codec_id = codec_.get_codec_id();
  timebase = codec_.get_time_base();
  crop = crop_;
  target_width = tgt_w;
  target_height = tgt_h;
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

  if (output_fmt != cudaVideoSurfaceFormat_NV12) {
    SPDL_FAIL(fmt::format(
        "Only NV12 output is supported. Found: {}",
        get_surface_format_name(output_fmt)));
  }

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

  CUstream stream = (CUstream)device_config.stream;
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
  auto width = decoder_param.ulTargetWidth;
  auto height = decoder_param.ulTargetHeight;

  if (decoder_param.OutputFormat != cudaVideoSurfaceFormat_NV12) {
    SPDL_FAIL(fmt::format(
        "Only NV12 is supported. Found: {}",
        get_surface_format_name(decoder_param.OutputFormat)));
  }
  auto h2 = height + height / 2;
  auto frame = std::make_shared<CUDAStorage>(width * h2, device_config);

  auto cfg = CUDA_MEMCPY2D{
      .srcXInBytes = 0,
      .srcY = 0,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcHost = nullptr,
      .srcDevice = (CUdeviceptr)mapping.frame,
      .srcArray = nullptr,
      .srcPitch = mapping.pitch,

      .dstXInBytes = 0,
      .dstY = 0,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstHost = nullptr,
      .dstDevice = (CUdeviceptr)(frame->data()),
      .dstArray = nullptr,
      .dstPitch = width,

      .WidthInBytes = width,
      .Height = h2,
  };
  TRACE_EVENT("nvdec", "cuMemcpy2DAsync");
  CHECK_CU(
      cuMemcpy2DAsync(&cfg, stream),
      "Failed to copy Y plane from decoder output surface.");
  CHECK_CU(cuStreamSynchronize(stream), "Failed to synchronize stream.");

  frame_buffer->emplace_back(CUDABuffer{
      device_config.device_index,
      frame,
      {h2, width},
      core::ElemClass::UInt,
      sizeof(uint8_t)});

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

void NvDecDecoderCore::decode_packet(
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

void NvDecDecoderCore::decode_packets(
    spdl::core::VideoPackets* packets,
    std::vector<CUDABuffer>* buffer) {
  if (device_config.device_index < 0) {
    SPDL_FAIL("Decoder is not initialized. Did you call `init`?");
  }
  TRACE_EVENT("nvdec", "decode_packets");

  // Init the temporary state used by the decoder callback during the decoding
  this->frame_buffer = buffer;
  if (packets->timestamp) {
    std::tie(start_time, end_time) = *(packets->timestamp);
  } else {
    start_time = -std::numeric_limits<double>::infinity();
    end_time = std::numeric_limits<double>::infinity();
  }

  auto ite = packets->iter_data();
  unsigned long flags = CUVID_PKT_TIMESTAMP;
  switch (codec_id) {
    case spdl::core::CodecID::MPEG4: {
      // TODO: Add special handling par
      // Video_Codec_SDK_12.1.14/blob/main/Samples/Utils/FFmpegDemuxer.h#L326-L345
      // TODO: Test this with MP4 file.
      SPDL_FAIL("NOT IMPLEMENTED.");
    }
    case spdl::core::CodecID::AV1: {
      // TODO handle
      // https://github.com/FFmpeg/FFmpeg/blob/5e2b0862eb1d408625232b37b7a2420403cd498f/libavcodec/cuviddec.c#L1001-L1009
      SPDL_FAIL("NOT IMPLEMENTED.");
    }
    default:;
  }

  flags |= CUVID_PKT_ENDOFPICTURE;
  while (ite) {
    auto pkt = ite();
    VLOG(9) << fmt::format("pkt.pts {}:", pkt.pts);
    decode_packet(pkt.data, pkt.size, pkt.pts, flags);
  }
}

void NvDecDecoderCore::flush(std::vector<CUDABuffer>* buffer) {
  this->frame_buffer = buffer;
  const unsigned char data{};
  decode_packet(&data, 0, 0, CUVID_PKT_ENDOFSTREAM);
}

void NvDecDecoderCore::reset() {
  if (parser) {
    cb_disabled = true;
    flush(nullptr);
    cb_disabled = false;
  }
}

} // namespace spdl::cuda::detail
