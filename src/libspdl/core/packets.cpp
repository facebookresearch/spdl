#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

extern "C" {
#include <libavcodec/avcodec.h>
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

namespace {
std::vector<std::tuple<size_t, size_t>> get_keyframe_indices(
    const std::vector<AVPacket*>& src_packets) {
  std::vector<std::tuple<size_t, size_t>> ret;
  size_t start = 0;
  for (size_t i = 1; i < src_packets.size(); ++i) {
    if (src_packets[i]->flags & AV_PKT_FLAG_KEY) {
      ret.push_back({start, i});
      start = i;
    }
  }
  ret.push_back({start, src_packets.size()});
  return ret;
}

VideoPacketsPtr
extract_packets(const VideoPacketsPtr& src, size_t start, size_t end) {
  auto& src_packets = src->get_packets();
  auto ret = std::make_unique<VideoPackets>(
      src->src, copy(src->codecpar), src->time_base);
  ret->timestamp = src->timestamp;
  ret->frame_rate = src->frame_rate;
  for (size_t t = start; t < end; ++t) {
    ret->push(CHECK_AVALLOCATE(av_packet_clone(src_packets[t])));
  }
  return ret;
}

} // namespace

std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    const std::vector<size_t>& indices) {
  auto& src_packets = src->get_packets();
  auto split_indices = get_keyframe_indices(src_packets);

  std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>> ret;
  size_t i = 0;
  for (auto& window : split_indices) {
    auto& [start, end] = window;

    std::vector<size_t> indices_in_window;
    while (i < indices.size() && (start <= indices[i] && indices[i] < end)) {
      indices_in_window.push_back(indices[i] - start);
      ++i;
    }
    if (indices_in_window.size() > 0) {
      ret.push_back({extract_packets(src, start, end), indices_in_window});
    }
    if (i >= indices.size()) {
      break;
    }
  }
  return ret;
}
} // namespace spdl::core
