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
    Rational time_base_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      timestamp(timestamp_),
      codecpar(copy(codecpar_)),
      time_base(time_base_) {
  TRACE_EVENT(
      "decoding",
      "DemuxedPackets::DemuxedPackets",
      perfetto::Flow::ProcessScoped(id));
};

template <MediaType media_type>
DemuxedPackets<media_type>::DemuxedPackets(
    std::string src_,
    AVCodecParameters* codecpar_,
    Rational time_base_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      codecpar(copy(codecpar_)),
      time_base(time_base_) {
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
PacketsPtr<media_type> clone(const DemuxedPackets<media_type>& src) {
  auto other = std::make_unique<DemuxedPackets<media_type>>(
      src.src, copy(src.codecpar), src.time_base);
  other->timestamp = src.timestamp;
  if constexpr (media_type == MediaType::Video) {
    other->frame_rate = src.frame_rate;
  }
  for (const AVPacket* src : src.get_packets()) {
    other->push(CHECK_AVALLOCATE(av_packet_clone(src)));
  }
  return other;
}

template PacketsPtr<MediaType::Audio> clone(const AudioPackets& src);
template PacketsPtr<MediaType::Video> clone(const VideoPackets& src);
template PacketsPtr<MediaType::Image> clone(const ImagePackets& src);

std::vector<VideoPacketsPtr> split_at_keyframes(const VideoPackets& src) {
  auto& src_packets = src.get_packets();
  // Search key frame indices
  std::vector<size_t> keyframe_indices;
  keyframe_indices.push_back(0); // always include the first frame
  for (size_t i = 1; i < src_packets.size(); ++i) {
    if (src_packets[i]->flags & AV_PKT_FLAG_KEY) {
      keyframe_indices.push_back(i);
    }
  }
  size_t num_keyframes = keyframe_indices.size();

  // Add the end to make the following operation simple
  keyframe_indices.push_back(src_packets.size());

  std::vector<VideoPacketsPtr> ret;
  ret.reserve(num_keyframes);
  for (size_t i = 0; i < num_keyframes; ++i) {
    auto start = keyframe_indices[i];
    auto end = keyframe_indices[i + 1];

    auto chunk = std::make_unique<VideoPackets>(
        src.src, copy(src.codecpar), src.time_base);
    chunk->timestamp = src.timestamp;
    chunk->frame_rate = src.frame_rate;
    for (size_t t = start; t < end; ++t) {
      chunk->push(CHECK_AVALLOCATE(av_packet_clone(src_packets[t])));
    }

    ret.push_back(std::move(chunk));
  }
  return ret;
}
} // namespace spdl::core
