#include <libspdl/core/detail/nvdec/decoding.h>

#include <libspdl/core/detail/cuda.h>
#include <libspdl/core/detail/logging.h>
#include <libspdl/core/detail/nvdec/decoder.h>
#include <libspdl/core/detail/nvdec/utils.h>
#include <libspdl/core/detail/nvdec/wrapper.h>
#include <libspdl/core/detail/tracing.h>
#include <libspdl/core/storage.h>

#include <folly/logging/xlog.h>

#include <cuda.h>
#include <cuviddec.h>

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

} // namespace

folly::coro::Task<std::unique_ptr<DecodedFrames>> decode_packets_nvdec(
    std::unique_ptr<PackagedAVPackets> packets,
    int cuda_device_index,
    const CropArea crop,
    int target_width,
    int target_height,
    const std::optional<std::string> pix_fmt,
    bool is_image) {
  size_t num_packets = packets->packets.size();
  assert(num_packets > 0);

  TRACE_EVENT("nvdec", "decode_packets");
  // Decoder objects are thread local so that they survive each decoding job,
  // which gives us opportunity to reuse the decoder and avoid destruction and
  // recreation ops, which are very expensive (~300ms).
  static thread_local NvDecDecoder decoder{};
  static thread_local bool decoding_ongoing = false;

  AVCodecParameters* codecpar = packets->codecpar;
  auto frames = std::make_unique<NvDecVideoFrames>(
      packets->id, MediaType::Video, codecpar->format);
  frames->buffer = std::make_shared<CUDABuffer2DPitch>(num_packets, is_image);

  if (decoding_ongoing) {
    // When the previous decoding ended with an error, if the new input data is
    // the same format, then handle_video_sequence might not be called because
    // NVDEC decoder thinks that incoming frames are from the same sequence.
    // This can cause strange decoder behavior. So we need to reset the decoder.
    decoder.reset();
  }
  decoder.init(
      cuda_device_index,
      covert_codec_id(codecpar->codec_id),
      frames->buffer.get(),
      packets->time_base,
      packets->timestamp,
      crop,
      target_width,
      target_height,
      pix_fmt);

  decoding_ongoing = true;
#define _PKT(i) packets->packets[i]
#define _PTS(pkt)                                           \
  (static_cast<double>(pkt->pts) * packets->time_base.num / \
   packets->time_base.den)

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

  flags |= CUVID_PKT_ENDOFPICTURE;
  for (; it < num_packets - 1; ++it) {
    auto pkt = _PKT(it);
    XLOG(DBG9) << fmt::format(
        " -- packet  PTS={:.3f} ({})", _PTS(pkt), pkt->pts);

    decoder.decode(pkt->data, pkt->size, pkt->pts, flags);
  }
  auto pkt = _PKT(it);
  flags |= CUVID_PKT_ENDOFSTREAM;
  decoder.decode(pkt->data, pkt->size, pkt->pts, flags);

  decoding_ongoing = false;

  XLOG(DBG5) << fmt::format(
      "Decoded {} frames from {} packets.",
      frames->buffer->n,
      packets->packets.size());

  co_return std::unique_ptr<DecodedFrames>(frames.release());
}

} // namespace spdl::core::detail

#undef _PKT
#undef _TS
