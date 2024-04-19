#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}
namespace spdl::core {
namespace {
inline AVCodecParameters* copy(const AVCodecParameters* src) {
  auto dst = CHECK_AVALLOCATE(avcodec_parameters_alloc());
  CHECK_AVERROR(
      avcodec_parameters_copy(dst, src), "Failed to copy codec parameters.");
  return dst;
}
} // namespace

template <MediaType media_type>
DemuxedPackets<media_type>::DemuxedPackets(
    std::string src_,
    std::tuple<double, double> timestamp_,
    AVCodecParameters* codecpar_,
    Rational time_base_,
    Rational frame_rate_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      timestamp(timestamp_),
      codecpar(copy(codecpar_)),
      time_base(time_base_),
      frame_rate(frame_rate_) {
  TRACE_EVENT(
      "decoding",
      "DemuxedPackets::DemuxedPackets",
      perfetto::Flow::ProcessScoped(id));
};

template <MediaType media_type>
DemuxedPackets<media_type>::~DemuxedPackets() {
  TRACE_EVENT(
      "decoding",
      "DemuxedPackets::~DemuxedPackets",
      perfetto::Flow::ProcessScoped(id));
  std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
  avcodec_parameters_free(&codecpar);
};

template <MediaType media_type>
void DemuxedPackets<media_type>::push(AVPacket* p) {
  if constexpr (media_type == MediaType::Image) {
    if (packets.size() > 0) {
      SPDL_FAIL_INTERNAL(
          "Multiple AVPacket is being pushed, but the expected number of AVPacket when decoding an image is one.");
    }
  }
  packets.push_back(p);
}

template <MediaType media_type>
size_t DemuxedPackets<media_type>::num_packets() const {
  return packets.size();
}

template <MediaType media_type>
const std::vector<AVPacket*>& DemuxedPackets<media_type>::get_packets() const {
  return packets;
}

template <MediaType media_type>
const char* DemuxedPackets<media_type>::get_media_format_name() const {
  return detail::get_media_format_name<media_type>(codecpar->format);
}

template struct DemuxedPackets<MediaType::Audio>;
template struct DemuxedPackets<MediaType::Video>;
template struct DemuxedPackets<MediaType::Image>;

template <MediaType media_type>
PacketsPtr<media_type> clone(const PacketsPtr<media_type>& src) {
  auto other = std::make_unique<DemuxedPackets<media_type>>(
      src->src,
      src->timestamp,
      copy(src->codecpar),
      src->time_base,
      src->frame_rate);
  for (const AVPacket* src : src->get_packets()) {
    other->push(CHECK_AVALLOCATE(av_packet_clone(src)));
  }
  return other;
}

template PacketsPtr<MediaType::Audio> clone(
    const PacketsPtr<MediaType::Audio>& src);
template PacketsPtr<MediaType::Video> clone(
    const PacketsPtr<MediaType::Video>& src);
template PacketsPtr<MediaType::Image> clone(
    const PacketsPtr<MediaType::Image>& src);

} // namespace spdl::core
