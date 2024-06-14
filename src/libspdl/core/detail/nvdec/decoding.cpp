#include "libspdl/core/detail/nvdec/decoding.h"

#include "libspdl/core/detail/cuda.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/nvdec/buffer.h"
#include "libspdl/core/detail/nvdec/decoder.h"
#include "libspdl/core/detail/nvdec/utils.h"
#include "libspdl/core/detail/nvdec/wrapper.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <cuda.h>
#include <cuviddec.h>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace spdl::core::detail {
namespace {

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

CUDABufferTracker get_buffer_tracker(
    const CUDAConfig& cuda_config,
    size_t num_packets,
    AVCodecParameters* codecpar,
    const CropArea& crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt,
    bool is_image) {
  size_t w = target_width > 0 ? target_width
                              : (codecpar->width - crop.left - crop.right);
  size_t h = target_height > 0 ? target_height
                               : (codecpar->height - crop.top - crop.bottom);

  size_t c;
  auto pix_fmt_val = pix_fmt.value_or("nv12");
  if (pix_fmt_val == "nv12") {
    c = 1;
    h = h + h / 2;
  } else if (pix_fmt_val == "rgba" || pix_fmt_val == "bgra") {
    c = 4;
  } else {
    SPDL_FAIL(fmt::format("Unsupported pixel format: {}", pix_fmt_val));
  }

  auto shape = is_image ? std::vector<size_t>{c, h, w}
                        : std::vector<size_t>{num_packets, c, h, w};

  return CUDABufferTracker{cuda_config, shape};
}

} // namespace

template <MediaType media_type>
CUDABufferPtr decode_nvdec(
    PacketsPtr<media_type> packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt) {
  size_t num_packets = packets->num_packets();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  ensure_cuda_initialized();

  TRACE_EVENT("nvdec", "decode_packets");

  _Decoder& _dec = get_decoder();

  AVCodecParameters* codecpar = packets->codecpar;
  auto tracker = get_buffer_tracker(
      cuda_config,
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
      cuda_config.device_index,
      covert_codec_id(codecpar->codec_id),
      &tracker,
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
    auto pkt = _PKT(it);
    VLOG(9) << fmt::format(" -- packet  PTS={:.3f} ({})", _PTS(pkt), pkt->pts);

    _dec.decoder.decode(pkt->data, pkt->size, pkt->pts, flags);
  }
  auto pkt = _PKT(it);
  flags |= CUVID_PKT_ENDOFSTREAM;
  _dec.decoder.decode(pkt->data, pkt->size, pkt->pts, flags);

  _dec.decoding_ongoing = false;

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
  return std::move(tracker.buffer);
}

template CUDABufferPtr decode_nvdec(
    VideoPacketsPtr packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt);

template CUDABufferPtr decode_nvdec(
    ImagePacketsPtr packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt);

CUDABufferPtr decode_nvdec(
    std::vector<ImagePacketsPtr>&& packets,
    const CUDAConfig& cuda_config,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string>& pix_fmt,
    bool strict) {
  size_t num_packets = packets.size();
  if (num_packets == 0) {
    SPDL_FAIL("No packets to decode.");
  }

  ensure_cuda_initialized();

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
        LOG(ERROR) << msg;
        packets.erase(packets.begin() + i);
      }
    }
  }

  TRACE_EVENT("nvdec", "decode_packets");

  _Decoder& _dec = get_decoder();

  auto tracker = get_buffer_tracker(
      cuda_config,
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
        cuda_config.device_index,
        covert_codec_id(packet->codecpar->codec_id),
        &tracker,
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
    int i_ = tracker.i;
    try {
      decode_fn(packet);
    } catch (std::exception& e) {
      LOG(ERROR) << fmt::format("Failed to decode image: {}", e.what());
      if (strict) {
        throw;
      }
    }
    // When a media is not supported, NVDEC does not necessarily
    // fail. Instead it just does not call the callback.
    // So we need to check if the number of decoded frames has changed.
    if (i_ == tracker.i) {
      auto msg =
          "Failed to decode image. (This could be due to unsupported media type. "
          "NVDEC does not always fail even when the input format is not supported.)";
      if (strict) {
        SPDL_FAIL(msg);
      } else {
        LOG(ERROR) << msg;
      }
    }
  }

  return std::move(tracker.buffer);
}

} // namespace spdl::core::detail
