/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <algorithm>
#include <cassert>

extern "C" {
#include <libavcodec/avcodec.h>
}
namespace spdl::core {

template <MediaType media_type>
DemuxedPackets<media_type>::DemuxedPackets(
    std::string src_,
    Codec<media_type>&& codec_,
    std::optional<std::tuple<double, double>> timestamp_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(std::move(src_)),
      timestamp(std::move(timestamp_)),
      codec(std::move(codec_)) {
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
        auto pts = static_cast<double>(pkt->pts) * codec.time_base.num /
            codec.time_base.den;
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
int64_t DemuxedPackets<media_type>::get_pts(size_t index) const {
  auto num_packets = packets.size();
  if (index >= num_packets) {
    throw std::out_of_range(
        fmt::format("{} is out of range [0, {})", index, num_packets));
  }
  return packets.at(index)->pts;
}

template <MediaType media_type>
int DemuxedPackets<media_type>::get_num_channels() const
  requires(media_type == MediaType::Audio)
{
  const auto* codecpar = codec.get_parameters();
  return GET_NUM_CHANNELS(codecpar);
}

template <MediaType media_type>
int DemuxedPackets<media_type>::get_sample_rate() const
  requires(media_type == MediaType::Audio)
{
  const auto* codecpar = codec.get_parameters();
  return codecpar->sample_rate;
}

template <MediaType media_type>
Codec<media_type> DemuxedPackets<media_type>::get_codec() const {
  return Codec<media_type>(
      codec.get_parameters(), codec.time_base, codec.frame_rate);
}

namespace {
template <MediaType media_type>
std::string get_codec_info(const AVCodecParameters* codecpar) {
  if (!codecpar) {
    return "<No codec information>";
  }

  std::vector<std::string> parts;

  parts.emplace_back(fmt::format("bit_rate={}", codecpar->bit_rate));
  parts.emplace_back(
      fmt::format("bits_per_sample={}", codecpar->bits_per_raw_sample));
  const AVCodecDescriptor* desc = avcodec_descriptor_get(codecpar->codec_id);
  parts.emplace_back(
      fmt::format("codec=\"{}\"", desc ? desc->name : "unknown"));

  if constexpr (media_type == MediaType::Audio) {
    parts.emplace_back(fmt::format("sample_rate={}", codecpar->sample_rate));
    parts.emplace_back(
        fmt::format("num_channels={}", GET_NUM_CHANNELS(codecpar)));
  }
  if constexpr (
      media_type == MediaType::Video || media_type == MediaType::Image) {
    parts.emplace_back(
        fmt::format("width={}, height={}", codecpar->width, codecpar->height));
  }
  return fmt::format("{}", fmt::join(parts, ", "));
}

std::string get_ts(const std::optional<std::tuple<double, double>>& ts) {
  return ts ? fmt::format("({}, {})", std::get<0>(*ts), std::get<1>(*ts))
            : "n/a";
}
} // namespace

template <>
std::string AudioPackets::get_summary() const {
  return fmt::format(
      "AudioPackets<src=\"{}\", timestamp={}, sample_format=\"{}\", {}>",
      src,
      get_ts(timestamp),
      get_media_format_name(),
      get_codec_info<MediaType::Audio>(codec.get_parameters()));
}

template <>
std::string VideoPackets::get_summary() const {
  return fmt::format(
      "VideoPackets<src=\"{}\", timestamp={}, frame_rate={}/{}, num_packets={}, pixel_format=\"{}\", {}>",
      src,
      get_ts(timestamp),
      codec.frame_rate.num,
      codec.frame_rate.den,
      num_packets(),
      get_media_format_name(),
      get_codec_info<MediaType::Video>(codec.get_parameters()));
}

template <>
std::string ImagePackets::get_summary() const {
  return fmt::format(
      "ImagePackets<src=\"{}\", pixel_format=\"{}\", {}>",
      src,
      get_media_format_name(),
      get_codec_info<MediaType::Image>(codec.get_parameters()));
}

template <MediaType media_type>
const std::vector<AVPacket*>& DemuxedPackets<media_type>::get_packets() const {
  return packets;
}

template <MediaType media_type>
const char* DemuxedPackets<media_type>::get_media_format_name() const {
  return detail::get_media_format_name<media_type>(
      codec.get_parameters()->format);
}

template <MediaType media_type>
int DemuxedPackets<media_type>::get_width() const {
  return codec.get_parameters()->width;
}

template <MediaType media_type>
int DemuxedPackets<media_type>::get_height() const {
  return codec.get_parameters()->height;
}

template <MediaType media_type>
Rational DemuxedPackets<media_type>::get_frame_rate() const {
  return codec.frame_rate;
}
template <MediaType media_type>
Generator<RawPacketData> DemuxedPackets<media_type>::iter_packets() const {
  for (auto& pkt : packets) {
    co_yield RawPacketData{pkt->data, pkt->size, pkt->pts};
  }
}

template class DemuxedPackets<MediaType::Audio>;
template class DemuxedPackets<MediaType::Video>;
template class DemuxedPackets<MediaType::Image>;

template <MediaType media_type>
PacketsPtr<media_type> clone(const DemuxedPackets<media_type>& src) {
  auto other = std::make_unique<DemuxedPackets<media_type>>(
      src.src, Codec<media_type>{src.codec}, src.timestamp);
  for (const AVPacket* pkt : src.get_packets()) {
    other->push(CHECK_AVALLOCATE(av_packet_clone(pkt)));
  }
  return other;
}

template PacketsPtr<MediaType::Audio> clone(const AudioPackets& src);
template PacketsPtr<MediaType::Video> clone(const VideoPackets& src);
template PacketsPtr<MediaType::Image> clone(const ImagePackets& src);

namespace {
std::vector<std::tuple<size_t, size_t, size_t>> get_keyframe_indices(
    const std::vector<AVPacket*>& packets) {
  // Split the input video packets into multiple sets of packets,
  // each of which starts with a key frame.
  //
  // Originally, the algorithm was simply splitting the packets at key frames,
  // but it turned out that there are video files that have packets of which
  // PTSs are below the PTS of corresponding key frames. Decoders, once receive
  // a key frame packet, discard packets with PTS below the PTS of the key frame
  // packets, so this resulted in discarding some frames.
  //
  // Therefore, in this split algorithm, packets are split in a way so that all
  // the non-key frame packets of which PTS are below the PTS of the key frame
  // of the next split are contained in the current split, along with the key
  // frame packets.
  //
  // For example, say we have packets with the following PTS.
  //
  // 1, 3, 4, 5,  2, 6, 8, 7
  // I        I  ^^^
  //
  // The PTS of the key frame of the second split is 5, but the packet with
  // PTS=2 comes after that. If we simply split at key frames, then PTS=2 will
  // be dicarded.
  //
  // 1, 3, 4,                -> 1, 3, 4
  // I
  //          5, 2, 6, 8, 7  -> 5, 6, 7, 8
  //          I
  //
  // So we split in a way that the stray packets are included in the previous
  // split.
  //
  // 1, 3, 4, 5, 2           -> 1, 2, 3, 4, 5
  //
  //          5, 2, 6, 8, 7  -> 5, 6, 7, 8
  //
  // The downside is that if we decode them as-is, the key frames are
  // duplicated. So care must be taken. The splitting algorithm is only used by
  // sample decoding, so we make sure that frames are not duplicated there.

  if (packets.size() == 0) {
    SPDL_FAIL("Packets cannot be empty");
  }

  // 1. Extract the PTS of key frames.
  std::vector<std::tuple<size_t, int64_t>> keyframe_pts;
  keyframe_pts.push_back({0, packets[0]->pts});
  for (size_t i = 1; i < packets.size(); ++i) {
    auto pkt = packets[i];
    if (pkt->flags & AV_PKT_FLAG_KEY) {
      keyframe_pts.push_back({i, pkt->pts});
    }
  }
  keyframe_pts.push_back({packets.size(), LLONG_MAX});

  // 2. Split the packets.
  // For N-th split, we extract the packets from the N-th key frame to the last
  // packet of which PTS is bellow the next split's key PTS.
  std::vector<std::tuple<size_t, size_t, size_t>> ret;
  ret.reserve(keyframe_pts.size() - 1);
  for (size_t split = 0; split < keyframe_pts.size() - 1; ++split) {
    auto [start, min_pts] = keyframe_pts[split];
    auto [end, max_pts] = keyframe_pts[split + 1];

    // Check if there are stray packets
    for (size_t i = end + 1; i < packets.size(); ++i) {
      auto pkt = packets[i];
      if (pkt->pts < max_pts) {
        end = i + 1;
      }
    }

    // obtain the number of invalid packets
    // invalid packets mean the PTS are less than min_pts.
    // Such packet should have been part of the previous split.
    size_t num_invalid = 0;
    for (size_t i = start; i < end; ++i) {
      if (packets[i]->pts < min_pts) {
        num_invalid += 1;
      }
    }
    ret.push_back({start, end, num_invalid});
  }
  return ret;
}

VideoPacketsPtr
extract_packets(const VideoPacketsPtr& src, size_t start, size_t end) {
  auto& src_packets = src->get_packets();
  auto ret = std::make_unique<VideoPackets>(
      src->src, VideoCodec{src->codec}, src->timestamp);
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
      auto pts = static_cast<double>(packet->pts) * src->codec.time_base.num /
          src->codec.time_base.den;
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
  for (auto& [start, end, num_invalid] : split_indices) {
    std::vector<size_t> indices_in_window;
    while (i < indices.size() && (start <= indices[i] && indices[i] < end)) {
      indices_in_window.push_back(indices[i] - start - num_invalid);
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
