/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/codec.h>

#include <c10/util/ScopeExit.h>
#include <libspdl/core/rational_utils.h>

#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include "libspdl/cuda/detail/utils.h"
#include "libspdl/cuda/nvdec/detail/decoder.h"
#include "libspdl/cuda/nvdec/detail/utils.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <sys/types.h>

#include <utility>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define CLOCKRATE 1

using spdl::core::is_within_window;
using spdl::core::to_double;
using spdl::core::to_rational;

namespace spdl::cuda::detail {
namespace {
class BoolRestorer {
  bool& target_;
  bool value_;

 public:
  BoolRestorer(bool& target, bool value) : target_{target}, value_{value} {}
  ~BoolRestorer() {
    target_ = value_;
  }

  BoolRestorer(const BoolRestorer&) = delete;
  BoolRestorer& operator=(const BoolRestorer&) = delete;
  BoolRestorer(BoolRestorer&&) = delete;
  BoolRestorer& operator=(BoolRestorer&&) = delete;
};

CUvideoctxlockPtr get_lock(CUcontext ctx) {
  CUvideoctxlock lock;
  CHECK_CU(cuvidCtxLockCreate(&lock, ctx), "Failed to create context lock.");
  return CUvideoctxlockPtr{lock};
}

int CUDAAPI video_sequence_callback(void* p, CUVIDEOFORMAT* data) noexcept {
  auto* core = static_cast<NvDecDecoderCore*>(p);
  return core->invoke_callback(
      [&]() { return core->handle_video_sequence(data); });
}

int CUDAAPI decode_picture_callback(void* p, CUVIDPICPARAMS* data) noexcept {
  auto* core = static_cast<NvDecDecoderCore*>(p);
  return core->invoke_callback(
      [&]() { return core->handle_decode_picture(data); });
}

int CUDAAPI
display_picture_callback(void* p, CUVIDPARSERDISPINFO* data) noexcept {
  auto* core = static_cast<NvDecDecoderCore*>(p);
  return core->invoke_callback(
      [&]() { return core->handle_display_picture(data); });
}

int CUDAAPI
operating_point_callback(void* p, CUVIDOPERATINGPOINTINFO* data) noexcept {
  auto* core = static_cast<NvDecDecoderCore*>(p);
  return core->invoke_callback(
      [&]() { return core->handle_operating_point(data); });
}

int CUDAAPI sei_message_callback(void* p, CUVIDSEIMESSAGEINFO* data) noexcept {
  auto* core = static_cast<NvDecDecoderCore*>(p);
  return core->invoke_callback([&]() { return core->handle_sei_msg(data); });
}

CUvideoparserPtr get_parser(
    NvDecDecoderCore* decoder,
    cudaVideoCodec codec_id,
    unsigned int max_num_decode_surfaces = 1,
    unsigned int max_display_delay = 2,
    bool extract_sei_message = true // temp
) {
  CUVIDPARSERPARAMS parser_params{
      .CodecType = codec_id,
      .ulMaxNumDecodeSurfaces = max_num_decode_surfaces,
      .ulClockRate = CLOCKRATE, // Timestamp units in Hz
      .ulMaxDisplayDelay = max_display_delay,
      .pUserData = (void*)decoder,
      .pfnSequenceCallback = video_sequence_callback,
      .pfnDecodePicture = decode_picture_callback,
      .pfnDisplayPicture = display_picture_callback,
      .pfnGetOperatingPoint = operating_point_callback,
      .pfnGetSEIMsg = extract_sei_message
          ? sei_message_callback
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
      return "Unknown decode status.";
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

NvDecDecoderCore::~NvDecDecoderCore() noexcept {
  if (cu_ctx_) {
    const CUresult status = push_current_(cu_ctx_);
    if (status != CUDA_SUCCESS) {
      LOG(WARNING) << fmt::format(
          "Failed to make the NVDEC context current during teardown ({}: {}); "
          "abandoning device resources to avoid teardown under the wrong "
          "context",
          get_error_name(status),
          get_error_desc(status));
      abandon_device_resources();
      return;
    }
  }
  release_device_resources();
  if (cu_ctx_) {
    CUcontext popped_context = nullptr;
    const CUresult status = cuCtxPopCurrent(&popped_context);
    if (status != CUDA_SUCCESS) {
      LOG(WARNING) << fmt::format(
          "Failed to restore the CUDA context after NVDEC teardown ({}: {})",
          get_error_name(status),
          get_error_desc(status));
    }
  }
}

void NvDecDecoderCore::abandon_device_resources() noexcept {
  (void)frame_buffer_.release();
  (void)parser_.release();
  (void)decoder_.release();
  (void)lock_.release();
  cap_cache_.clear();
}

void NvDecDecoderCore::release_device_resources() noexcept {
  frame_buffer_.reset();
  parser_.reset();
  decoder_.reset();
  lock_.reset();
  cap_cache_.clear();
}

void NvDecDecoderCore::init_decoder(
    const CUDAConfig& device_config,
    const spdl::core::VideoCodec& codec,
    const CropArea& crop,
    int tgt_w,
    int tgt_h) {
  if (auto tb = codec.get_time_base(); tb.num <= 0 || tb.den <= 0) {
    SPDL_FAIL_INTERNAL(
        fmt::format("Invalid time base was found: {}/{}", tb.num, tb.den));
  }
  if (crop.left < 0) {
    SPDL_FAIL(
        fmt::format("crop_left must be non-negative. Found: {}", crop.left));
  }
  if (crop.top < 0) {
    SPDL_FAIL(
        fmt::format("crop_top must be non-negative. Found: {}", crop.top));
  }
  if (crop.right < 0) {
    SPDL_FAIL(
        fmt::format("crop_right must be non-negative. Found: {}", crop.right));
  }
  if (crop.bottom < 0) {
    SPDL_FAIL(
        fmt::format(
            "crop_bottom must be non-negative. Found: {}", crop.bottom));
  }
  if (tgt_w > 0 && tgt_w % 2) {
    SPDL_FAIL(fmt::format("target_width must be positive. Found: {}", tgt_w));
  }
  if (tgt_h > 0 && tgt_h % 2) {
    SPDL_FAIL(fmt::format("target_height must be positive. Found: {}", tgt_h));
  }
  const bool switching_device =
      device_config_.device_index != device_config.device_index;
  CUcontext calling_ctx = nullptr;
  if (switching_device) {
    CHECK_CU(
        cuCtxGetCurrent(&calling_ctx), "Failed to get current CUDA context.");
  }
  auto restore_context_on_error =
      c10::make_scope_exit([this, switching_device, calling_ctx]() noexcept {
        if (!switching_device) {
          return;
        }
        // Ensure a partially initialized device can never be mistaken for a
        // reusable decoder, even if copying CUDAConfig below throws midway.
        device_config_.device_index = -1;
        const CUresult status = cuCtxSetCurrent(calling_ctx);
        if (status != CUDA_SUCCESS) {
          LOG(WARNING) << fmt::format(
              "Failed to restore the CUDA context after NVDEC initialization "
              "failed ({}: {})",
              get_error_name(status),
              get_error_desc(status));
        }
      });
  if (switching_device) {
    const CUcontext next_ctx = get_cucontext(device_config.device_index);

    // The parser and decoder retain the context lock, and all device-owned
    // resources must be released while their original context is current.
    if (cu_ctx_) {
      CHECK_CU(cuCtxSetCurrent(cu_ctx_), "Failed to set current context.");
    }
    release_device_resources();
    // From this point onward the previous context no longer owns any live
    // decoder resources. Invalidate the cached device state before operations
    // on the replacement context can throw, so a later retry rebuilds both the
    // context lock and resources instead of treating this partial transition as
    // an initialized decoder.
    cu_ctx_ = nullptr;
    device_config_.device_index = -1;

    CHECK_CU(cuCtxSetCurrent(next_ctx), "Failed to set current context.");
    auto next_lock = get_lock(next_ctx);
    cu_ctx_ = next_ctx;
    lock_ = std::move(next_lock);
  }
  auto cdc = convert_codec_id(codec.get_codec_id());
  if (!parser_ || codec_ != cdc) {
    VLOG(9) << "initializing parser";
    codec_ = cdc;
    parser_ = get_parser(this, codec_);
    decoder_ = nullptr;
    decoder_param_.ulMaxHeight = 720;
    decoder_param_.ulMaxWidth = 1280;
  }

  src_width_ = codec.get_width();
  src_height_ = codec.get_height();
  codec_id_ = codec.get_codec_id();
  timebase_ = codec.get_time_base();
  crop_ = crop;
  target_width_ = tgt_w;
  target_height_ = tgt_h;

  // Reset frame buffer for new stream
  frame_buffer_.reset();
  time_window_ = std::nullopt;

  // Commit the cached configuration only after every throwing initialization
  // step succeeds. A failed device switch must remain retryable.
  device_config_ = device_config;

  // Later NVDEC calls and their synchronous callbacks use the context selected
  // here. Preserve that success-path contract, but restore the caller's context
  // when initialization exits through an exception.
  restore_context_on_error.release();
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
  if (cb_disabled_) {
    return 1;
  }
  TRACE_EVENT("nvdec", "NvDecDecoderCore::handle_video_sequence");

  VLOG(9) << print(video_fmt);

  // Check if the input video is supported.
  CUVIDDECODECAPS caps = check_capacity(video_fmt, cap_cache_);
  auto output_fmt = get_output_sufrace_format(video_fmt, &caps);

  if (output_fmt != cudaVideoSurfaceFormat_NV12) {
    SPDL_FAIL(
        fmt::format(
            "Only NV12 output is supported. Found: {}",
            get_surface_format_name(output_fmt)));
  }

  unsigned long max_width =
      MAX(video_fmt->coded_width, decoder_param_.ulMaxWidth);
  unsigned long max_height =
      MAX(video_fmt->coded_height, decoder_param_.ulMaxHeight);

  // Get parameters for creating decoder.
  auto new_decoder_param = get_create_info(
      reinterpret_cast<CUvideoctxlock>(lock_.get()),
      video_fmt,
      output_fmt,
      max_width,
      max_height,
      crop_,
      target_width_,
      target_height_);

  VLOG(5) << print(&new_decoder_param);

  // Update decoder
  auto ret = [&]() -> unsigned long {
    if (!decoder_) {
      decoder_.reset(get_decoder(&new_decoder_param));
      return new_decoder_param.ulNumDecodeSurfaces;
    }
    switch (update_type(decoder_param_, new_decoder_param)) {
      case RETAIN:
        break;
      case RECONFIGURE:
        reconfigure_decoder(decoder_.get(), new_decoder_param);
        break;
      case RECREATE:
        decoder_.reset(get_decoder(&new_decoder_param));
        break;
    }
    auto prev_num_surfs = decoder_param_.ulNumDecodeSurfaces;
    return prev_num_surfs == new_decoder_param.ulNumDecodeSurfaces
        ? 1
        : new_decoder_param.ulNumDecodeSurfaces;
  }();
  decoder_param_ = new_decoder_param;

  return (int)ret;
}

int NvDecDecoderCore::handle_decode_picture(CUVIDPICPARAMS* pic_params) {
  // This function is called by the parser when the input bit stream is parsed
  // and ready for decodings It just kicks off the decoding work.
  //
  // Return values
  // * 0: fail
  // * >=1: success

  if (cb_disabled_) {
    return 1;
  }
  TRACE_EVENT("nvdec", "NvDecDecoderCore::handle_decode_picture");

  // LOG(INFO) << "Received decoded pictures.";
  // LOG(INFO) << print(pic_params);
  if (!decoder_) {
    SPDL_FAIL_INTERNAL("Decoder not initialized.");
  }
  TRACE_EVENT("nvdec", "cuvidDecodePicture");
  CHECK_CU(
      cuvidDecodePicture(decoder_.get(), pic_params),
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
  // * >=1: success

  // LOG(INFO) << "Received display pictures.";
  // LOG(INFO) << print(disp_info);

  if (cb_disabled_) {
    return 1;
  }
  TRACE_EVENT("nvdec", "NvDecDecoderCore::handle_display_picture");

  auto pts = to_rational(disp_info->timestamp, timebase_);

  VLOG(9) << fmt::format(
      " --- Frame  PTS={:.3f} ({})", to_double(pts), disp_info->timestamp);

  if (time_window_) {
    auto [s, t] = *time_window_;
    if (!is_within_window(pts, s, t)) {
      return 1;
    }
  }

  if (!frame_buffer_) {
    SPDL_FAIL_INTERNAL("FrameBuffer not initialized.");
  }

  auto width = decoder_param_.ulTargetWidth;
  auto height = decoder_param_.ulTargetHeight;

  VLOG(9) << fmt::format("{} x {}", width, height);

  if (decoder_param_.OutputFormat != cudaVideoSurfaceFormat_NV12) {
    SPDL_FAIL(
        fmt::format(
            "Only NV12 is supported. Found: {}",
            get_surface_format_name(decoder_param_.OutputFormat)));
  }

  warn_if_error(decoder_.get(), disp_info->picture_index);

  CUVIDPROCPARAMS proc_params{
      .progressive_frame = disp_info->progressive_frame,
      .second_field = disp_info->repeat_first_field + 1,
      .top_field_first = disp_info->top_field_first,
      .unpaired_field = disp_info->repeat_first_field < 0,
      .output_stream = (CUstream)device_config_.stream};

  // Make the decoded frame available to output surface
  MapGuard mapping(decoder_.get(), &proc_params, disp_info->picture_index);

  // Push frame to FrameBuffer
  frame_buffer_->push((void*)mapping.frame, mapping.pitch);

  return 1;
}

int NvDecDecoderCore::handle_operating_point(CUVIDOPERATINGPOINTINFO*) {
  // Return values:
  // * <0: fail
  // * >=0: success
  //    - bit 0-9: OperatingPoint
  //    - bit 10-10: outputAllLayers
  //    - bit 11-30: reserved

  if (cb_disabled_) {
    return 1;
  }
  TRACE_EVENT("nvdec", "NvDecDecoderCore::handle_operating_point");

  // LOG(INFO) << "Received operating points.";

  // Not implemented yet.
  return 0;
}

int NvDecDecoderCore::handle_sei_msg(CUVIDSEIMESSAGEINFO*) {
  // Return values:
  // * 0: fail
  // * >=1: succeeded

  if (cb_disabled_) {
    return 1;
  }

  // LOG(INFO) << "Received SEI messages.";
  // LOG(INFO) << print(msg_info);
  return 0;
}

void NvDecDecoderCore::decode_packet(
    const spdl::core::RawPacketData& pkt,
    unsigned long flags) {
  VLOG(9) << fmt::format("pkt.pts {}:", pkt.pts);
  switch (codec_id_) {
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
  // TODO: Turn these check into debug-only assertions. because they are only
  // used by `NvDecDecoder`.
  if (device_config_.device_index < 0) {
    SPDL_FAIL("Decoder is not initialized. Did you call `init_decoder`?");
  }
  if (!cb_disabled_ && !frame_buffer_) {
    SPDL_FAIL_INTERNAL(
        "Frame buffer is not initialized. Did you call `init_buffer`?");
  }
  if (!parser_) {
    SPDL_FAIL_INTERNAL("Parser is not initialized.");
  }

  TRACE_EVENT("nvdec", "NvDecDecoderCore::decode_packet");

  CUVIDSOURCEDATAPACKET packet{
      .flags = flags,
      .payload_size = static_cast<unsigned long>(pkt.size),
      .payload = pkt.data,
      .timestamp = pkt.pts};

  callback_error_ = nullptr;
  const CUresult status = cuvidParseVideoData(parser_.get(), &packet);
  rethrow_callback_error();
  CHECK_CU(status, "Failed to parse video data.");
}

void NvDecDecoderCore::rethrow_callback_error() {
  if (auto error = std::exchange(callback_error_, std::exception_ptr{})) {
    std::rethrow_exception(error);
  }
}

void NvDecDecoderCore::reset() {
  if (parser_) {
    const bool callbacks_were_disabled = std::exchange(cb_disabled_, true);
    const BoolRestorer restore_callbacks{cb_disabled_, callbacks_were_disabled};
    flush();
  }
}

void NvDecDecoderCore::init_buffer(size_t num_frames) {
  int width =
      target_width_ > 0 ? target_width_ : src_width_ - crop_.left - crop_.right;
  int height = target_height_ > 0 ? target_height_
                                  : src_height_ - crop_.top - crop_.bottom;

  frame_buffer_ =
      std::make_unique<FrameBuffer>(num_frames, width, height, device_config_);
}

bool NvDecDecoderCore::has_batch_ready() const {
  return frame_buffer_ && !frame_buffer_->empty();
}

CUDABuffer NvDecDecoderCore::pop_batch() {
  if (!frame_buffer_ || frame_buffer_->empty()) {
    SPDL_FAIL_INTERNAL("No batch ready to pop");
  }
  return std::move(*frame_buffer_->pop());
}

void NvDecDecoderCore::flush() {
  if (device_config_.device_index < 0) {
    SPDL_FAIL("Decoder is not initialized. Did you call `init_decoder`?");
  }
  TRACE_EVENT("nvdec", "NvDecDecoderCore::flush");

  // Flush decoder by sending empty packet with ENDOFSTREAM flag
  unsigned char data{};
  decode_packet({&data, 0, 0}, CUVID_PKT_ENDOFSTREAM);

  if (frame_buffer_) {
    frame_buffer_->flush();
  }
}

CUDABuffer NvDecDecoderCore::decode_packets(spdl::core::VideoPackets* packets) {
  if (device_config_.device_index < 0) {
    SPDL_FAIL("Decoder is not initialized. Did you call `init_decoder`?");
  }
  TRACE_EVENT("nvdec", "NvDecDecoderCore::decode_packets");

  size_t max_frames = packets->pkts.get_packets().size();

  // Set up FrameBuffer for batch mode
  init_buffer(max_frames);
  time_window_ = packets->timestamp;

  for (auto pkt : packets->pkts.iter_data()) {
    decode_packet(pkt);
  }
  flush();

  // Retrieve the buffer from FrameBuffer
  if (frame_buffer_->empty()) {
    SPDL_FAIL_INTERNAL("No frames were decoded.");
  }
  CUDABufferPtr ret_ptr = frame_buffer_->pop();

  // Reset batch mode state
  frame_buffer_.reset();

  return std::move(*ret_ptr);
}

} // namespace spdl::cuda::detail
