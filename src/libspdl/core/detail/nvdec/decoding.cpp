#include <libspdl/core/detail/nvdec/decoding.h>

#include <libspdl/core/storage.h>

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/nvdec/decoder.h"
#include "libspdl/core/detail/nvdec/utils.h"
#include "libspdl/core/detail/nvdec/wrapper.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

#include <cuda.h>
#include <cuviddec.h>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace spdl::core::detail {
namespace {
CUstream get_stream() {
  TRACE_EVENT("nvdec", "cuStreamCreate");
  CUstream stream;
  CHECK_CU(
      cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), // CU_STREAM_DEFAULT
      "Failed to create stream.");
  XLOG(DBG5) << "CUDA stream: " << (void*)stream;
  return stream;
}

struct _Decoder {
  NvDecDecoder decoder{};
  bool decoding_ongoing = false;
};

_Decoder& get_decoder() {
  // Decoder objects are thread local so that they survive each decoding job,
  // which gives us opportunity to reuse the decoder and avoid destruction and
  // recreation ops, which are very expensive (~300ms).
  static thread_local _Decoder decoder{};
  return decoder;
}

std::shared_ptr<CUDABuffer2DPitch> get_buffer(
    int cuda_device_index,
    size_t num_packets,
    AVCodecParameters* codecpar,
    const CropArea& crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt,
    bool is_image) {
  int w = target_width > 0 ? target_width
                           : (codecpar->width - crop.left - crop.right);
  int h = target_height > 0 ? target_height
                            : (codecpar->height - crop.top - crop.bottom);

  auto cu_ctx = get_cucontext(cuda_device_index);
  CHECK_CU(cuCtxSetCurrent(cu_ctx), "Failed to set current context.");

  if (!pix_fmt) { // Assume NV12
    return std::make_shared<CUDABuffer2DPitch>(
        cuda_device_index, num_packets, 1, h + h / 2, w, is_image);
  }
  auto pix_fmt_val = pix_fmt.value();
  if (pix_fmt_val == "rgba" || pix_fmt_val == "bgra") {
    return std::make_shared<CUDABuffer2DPitch>(
        cuda_device_index, num_packets, 4, h, w, is_image);
  }
  SPDL_FAIL(fmt::format(
      "Unsupported pixel format: {}. Supported formats are 'rgba', 'bgra'.",
      pix_fmt_val));
}

} // namespace

template <MediaType media_type>
folly::coro::Task<NvDecFramesPtr<media_type>> decode_nvdec(
    PacketsPtr<media_type> packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string> pix_fmt) {
  co_await folly::coro::co_safe_point;
  size_t num_packets = packets->num_packets();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  TRACE_EVENT("nvdec", "decode_packets");

  _Decoder& _dec = get_decoder();

  AVCodecParameters* codecpar = packets->codecpar;
  auto frames = std::make_unique<NvDecFrames<media_type>>(
      packets->id,
      pix_fmt ? av_get_pix_fmt(pix_fmt->c_str()) : codecpar->format);

  frames->buffer = get_buffer(
      cuda_device_index,
      num_packets,
      codecpar,
      crop,
      target_width,
      target_height,
      pix_fmt,
      media_type == MediaType::Image);

  if (_dec.decoding_ongoing) {
    // When the previous decoding ended with an error, if the new input data is
    // the same format, then handle_video_sequence might not be called because
    // NVDEC decoder thinks that incoming frames are from the same sequence.
    // This can cause strange decoder behavior. So we need to reset the decoder.
    _dec.decoder.reset();
  }
  _dec.decoder.init(
      cuda_device_index,
      covert_codec_id(codecpar->codec_id),
      frames->buffer.get(),
      packets->time_base,
      packets->timestamp,
      crop,
      target_width,
      target_height,
      pix_fmt);

  _dec.decoding_ongoing = true;
  size_t it = 0;
  unsigned long flags = CUVID_PKT_TIMESTAMP;
  switch (codecpar->codec_id) {
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
    co_await folly::coro::co_safe_point;
    auto pkt = _PKT(it);
    XLOG(DBG9) << fmt::format(
        " -- packet  PTS={:.3f} ({})", _PTS(pkt), pkt->pts);

    _dec.decoder.decode(pkt->data, pkt->size, pkt->pts, flags);
  }
  auto pkt = _PKT(it);
  flags |= CUVID_PKT_ENDOFSTREAM;
  _dec.decoder.decode(pkt->data, pkt->size, pkt->pts, flags);

  _dec.decoding_ongoing = false;

#undef _PKT
#undef _PTS

  XLOG(DBG5) << fmt::format(
      "Decoded {} frames from {} packets.",
      frames->buffer->n,
      packets->num_packets());

  co_return frames;
}

template folly::coro::Task<NvDecVideoFramesPtr> decode_nvdec(
    VideoPacketsPtr packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string> pix_fmt);

template folly::coro::Task<NvDecImageFramesPtr> decode_nvdec(
    ImagePacketsPtr packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string> pix_fmt);

folly::coro::Task<NvDecVideoFramesPtr> decode_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string> pix_fmt,
    bool strict) {
  co_await folly::coro::co_safe_point;
  size_t num_packets = packets.size();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  auto& p0 = packets[0];
  if (!pix_fmt) {
    for (auto& packet : packets) {
      if (packet->codecpar->format != p0->codecpar->format) {
        SPDL_FAIL(fmt::format(
            "All images must have the same pixel format. The first one has {}, while another has {}.",
            av_get_pix_fmt_name((AVPixelFormat)p0->codecpar->format),
            av_get_pix_fmt_name((AVPixelFormat)packet->codecpar->format)));
      }
    }
  }
  for (int i = num_packets - 1; i >= 0; --i) {
    // An example of YUVJ411P format is "ILSVRC2012_val_00016550.JPEG" from
    // ImageNet validation set.
    // When such a data is fed to NVDEC, it just does not call the callback.
    if ((AVPixelFormat)packets[i]->codecpar->format == AV_PIX_FMT_YUVJ411P) {
      auto msg =
          fmt::format("Found unsupported format YUVJ411P at index {}.", i);
      if (strict) {
        SPDL_FAIL(msg);
      } else {
        XLOG(ERR) << msg;
        packets.erase(packets.begin() + i);
      }
    }
  }

  TRACE_EVENT("nvdec", "decode_packets");

  _Decoder& _dec = get_decoder();

  auto frames = std::make_unique<NvDecVideoFrames>(
      packets[0]->id,
      pix_fmt ? av_get_pix_fmt(pix_fmt->c_str()) : p0->codecpar->format);

  frames->buffer = get_buffer(
      cuda_device_index,
      num_packets,
      p0->codecpar,
      crop,
      target_width,
      target_height,
      pix_fmt,
      false);

  auto decode_fn = [&](ImagePacketsPtr& packet) {
    if (_dec.decoding_ongoing) {
      // When the previous decoding ended with an error, if the new input data
      // is the same format, then handle_video_sequence might not be called
      // because NVDEC decoder thinks that incoming frames are from the same
      // sequence. This can cause strange decoder behavior. So we need to reset
      // the decoder.
      _dec.decoder.reset();
    }
    _dec.decoder.init(
        cuda_device_index,
        covert_codec_id(packet->codecpar->codec_id),
        frames->buffer.get(),
        packet->time_base,
        packet->timestamp,
        crop,
        target_width,
        target_height,
        pix_fmt);

    _dec.decoding_ongoing = true;
    unsigned long flags =
        CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE | CUVID_PKT_ENDOFSTREAM;

    auto pkt = packet->get_packets()[0];
    _dec.decoder.decode(pkt->data, pkt->size, pkt->pts, flags);
    _dec.decoding_ongoing = false;
  };

  for (auto& packet : packets) {
    co_await folly::coro::co_safe_point;
    int n = frames->buffer->n;
    try {
      decode_fn(packet);
    } catch (std::exception& e) {
      XLOG(ERR) << fmt::format("Failed to decode image: {}", e.what());
      if (strict) {
        throw;
      }
    }
    // When a media is not supported, NVDEC does not necessarily
    // fail. Instead it just does not call the callback.
    // So we need to check if the number of decoded frames has changed.
    if (n == frames->buffer->n) {
      if (strict) {
        SPDL_FAIL(
            "Failed to decode image. "
            "(This could be due to unsupported media type. "
            "NVDEC does not always fail even when the input format is not supported.)");
      } else {
        XLOG(ERR)
            << "Failed to decode image. "
               "(This could be due to unsupported media type. "
               "NVDEC does not always fail even when the input format is not supported.)";
      }
    }
  }
  co_return frames;
}

} // namespace spdl::core::detail
