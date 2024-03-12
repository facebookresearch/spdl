#include "libspdl/core/detail/ffmpeg/package.h"

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/logging/xlog.h>

#include <random>

namespace spdl::core::detail {
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

PackagedAVPackets::PackagedAVPackets(
    MediaType media_type_,
    std::string src_,
    std::tuple<double, double> timestamp_,
    AVCodecParameters* codecpar_,
    AVRational time_base_,
    AVRational frame_rate_)
    : id(random()),
      media_type(media_type_),
      src(src_),
      timestamp(timestamp_),
      codecpar(copy(codecpar_)),
      time_base(time_base_),
      frame_rate(frame_rate_) {
  TRACE_EVENT(
      "decoding",
      "PackagedAVPackets::PackagedAVPackets",
      perfetto::Flow::ProcessScoped(id));
};

PackagedAVPackets::PackagedAVPackets(PackagedAVPackets&& other) noexcept {
  *this = std::move(other);
};

PackagedAVPackets& PackagedAVPackets::operator=(
    PackagedAVPackets&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(media_type, other.media_type);
  swap(src, other.src);
  swap(timestamp, other.timestamp);
  swap(codecpar, other.codecpar);
  swap(time_base, other.time_base);
  swap(frame_rate, other.frame_rate);
  swap(packets, other.packets);
  return *this;
};

PackagedAVPackets::~PackagedAVPackets() {
  TRACE_EVENT(
      "decoding",
      "PackagedAVPackets::~PackagedAVPackets",
      perfetto::Flow::ProcessScoped(id));
  std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
  avcodec_parameters_free(&codecpar);
};

} // namespace spdl::core::detail
