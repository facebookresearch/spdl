#include <libspdl/detail/ffmpeg/logging.h>
#include <libspdl/detail/packets.h>

namespace spdl::detail {
namespace {
inline AVCodecParameters* copy(const AVCodecParameters* src) {
  auto dst = CHECK_AVALLOCATE(avcodec_parameters_alloc());
  CHECK_AVERROR(
      avcodec_parameters_copy(dst, src), "Failed to copy codec parameters.");
  return dst;
}
} // namespace

PackagedAVPackets::PackagedAVPackets(
    std::string src_,
    std::tuple<double, double> timestamp_,
    AVCodecParameters* codecpar_,
    AVRational time_base_,
    AVRational frame_rate_)
    : src(src_),
      timestamp(timestamp_),
      codecpar(copy(codecpar_)),
      time_base(time_base_),
      frame_rate(frame_rate_){};

PackagedAVPackets::PackagedAVPackets(PackagedAVPackets&& other) noexcept {
  *this = std::move(other);
};

PackagedAVPackets& PackagedAVPackets::operator=(
    PackagedAVPackets&& other) noexcept {
  using std::swap;
  swap(src, other.src);
  swap(timestamp, other.timestamp);
  swap(codecpar, other.codecpar);
  swap(time_base, other.time_base);
  swap(frame_rate, other.frame_rate);
  swap(packets, other.packets);
  return *this;
};

PackagedAVPackets::~PackagedAVPackets() {
  std::for_each(packets.begin(), packets.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
  avcodec_parameters_free(&codecpar);
};

} // namespace spdl::detail
