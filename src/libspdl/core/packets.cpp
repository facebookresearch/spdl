#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

#include <random>

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

uint64_t random() {
  static thread_local std::random_device rd;
  static thread_local std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  return dis(gen);
}
} // namespace

template <MediaType media_type>
DemuxedPackets<media_type>::DemuxedPackets(
    std::string src_,
    std::tuple<double, double> timestamp_,
    AVCodecParameters* codecpar_,
    Rational time_base_,
    Rational frame_rate_)
    : id(random()),
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
DemuxedPackets<media_type>::DemuxedPackets(
    DemuxedPackets<media_type>&& other) noexcept {
  *this = std::move(other);
};

template <MediaType media_type>
DemuxedPackets<media_type>& DemuxedPackets<media_type>::operator=(
    DemuxedPackets<media_type>&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(src, other.src);
  swap(timestamp, other.timestamp);
  swap(codecpar, other.codecpar);
  swap(time_base, other.time_base);
  swap(frame_rate, other.frame_rate);
  swap(packets, other.packets);
  return *this;
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

template struct DemuxedPackets<MediaType::Audio>;
template struct DemuxedPackets<MediaType::Video>;
template struct DemuxedPackets<MediaType::Image>;

} // namespace spdl::core
