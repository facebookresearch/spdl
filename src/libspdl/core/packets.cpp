/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <glog/logging.h>

#include <cassert>

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
  if (!p) {
    SPDL_FAIL_INTERNAL("Packet is NULL.");
  }
  packets.push_back(p);
}

template <MediaType media_type>
size_t DemuxedPackets<media_type>::num_packets() const
  requires(media_type == MediaType::Video || media_type == MediaType::Image)
{
  if constexpr (media_type == MediaType::Image) {
    assert(packets.size() == 1);
    return 1;
  }
  if constexpr (media_type == MediaType::Video) {
    if (timestamp) {
      size_t ret = 0;
      auto [start, end] = *timestamp;
      for (const AVPacket* pkt : packets) {
        auto pts =
            static_cast<double>(pkt->pts) * time_base.num / time_base.den;
        if (start <= pts && pts < end) {
          ++ret;
        }
      }
      return ret;
    }
    return packets.size();
  }
}

template <MediaType media_type>
const std::vector<AVPacket*>& DemuxedPackets<media_type>::get_packets() const {
  return packets;
}

template <MediaType media_type>
const char* DemuxedPackets<media_type>::get_media_format_name() const {
  return detail::get_media_format_name<media_type>(codecpar->format);
}

template <MediaType media_type>
int DemuxedPackets<media_type>::get_width() const {
  assert(codecpar);
  return codecpar->width;
}

template <MediaType media_type>
int DemuxedPackets<media_type>::get_height() const {
  assert(codecpar);
  return codecpar->height;
}

template <MediaType media_type>
Rational DemuxedPackets<media_type>::get_frame_rate() const {
  return frame_rate;
}

template class DemuxedPackets<MediaType::Audio>;
template class DemuxedPackets<MediaType::Video>;
template class DemuxedPackets<MediaType::Image>;

template <MediaType media_type>
PacketsPtr<media_type> clone(const DemuxedPackets<media_type>& src) {
  auto other = std::make_unique<DemuxedPackets<media_type>>(
      src.src, copy(src.codecpar), src.time_base);
  other->timestamp = src.timestamp;
  if constexpr (media_type == MediaType::Video) {
    other->frame_rate = src.frame_rate;
  }
  for (const AVPacket* pkt : src.get_packets()) {
    other->push(CHECK_AVALLOCATE(av_packet_clone(pkt)));
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
    std::vector<size_t> indices) {
  auto& src_packets = src->get_packets();
  // If timestamp is set, then there are frames before the window.
  // `indices` are supposed to be within the window.
  // So we adjust the `indices` by shifitng the number of frames before the
  // window.
  if (src->timestamp) {
    auto [start, end] = *(src->timestamp);
    size_t offset = 0;
    for (auto& packet : src_packets) {
      auto pts = static_cast<double>(packet->pts) * src->time_base.num /
          src->time_base.den;
      if (pts < start) {
        offset += 1;
      }
    }
    for (size_t& i : indices) {
      i += offset;
    }
  }
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
