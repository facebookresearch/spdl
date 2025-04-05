/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <glog/logging.h>

#include <algorithm>
#include <cassert>

extern "C" {
#include <libavcodec/avcodec.h>
}
namespace spdl::core {

PacketSeries::PacketSeries(){};

PacketSeries::PacketSeries(const PacketSeries& other) {
  for (const AVPacket* pkt : other.container) {
    container.push_back(CHECK_AVALLOCATE(av_packet_clone(pkt)));
  }
};

PacketSeries::PacketSeries(PacketSeries&& other) noexcept {
  *this = std::move(other);
};

PacketSeries& PacketSeries::operator=(const PacketSeries& other) {
  PacketSeries tmp(other);
  *this = std::move(tmp);
  return *this;
};

PacketSeries& PacketSeries::operator=(PacketSeries&& other) noexcept {
  using std::swap;
  swap(container, other.container);
  return *this;
};

PacketSeries::~PacketSeries() {
  std::for_each(container.begin(), container.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
};

void PacketSeries::push(AVPacket* p) {
  if (!p) {
    SPDL_FAIL_INTERNAL("Packet is NULL.");
  }
  container.push_back(p);
}

template <MediaType media>
Packets<media>::Packets(
    const std::string& src_,
    Codec<media>&& codec_,
    const std::optional<std::tuple<double, double>>& timestamp_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      time_base(codec_.get_time_base()),
      timestamp(timestamp_),
      codec(std::move(codec_)) {
  TRACE_EVENT(
      "decoding", "Packets::Packets", perfetto::Flow::ProcessScoped(id));
};

template <MediaType media>
Packets<media>::Packets(const std::string& src_, const Rational& time_base_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      time_base(time_base_),
      timestamp(std::nullopt),
      codec(std::nullopt) {
  TRACE_EVENT(
      "decoding", "Packets::Packets", perfetto::Flow::ProcessScoped(id));
};
template <MediaType media>
Packets<media>::Packets(uintptr_t id_, Rational time_base_)
    : id(id_),
      src(fmt::format("{}", id_)),
      time_base(time_base_),
      timestamp(std::nullopt),
      codec(std::nullopt) {
  TRACE_EVENT(
      "decoding", "Packets::Packets", perfetto::Flow::ProcessScoped(id));
};

template <MediaType media>
Packets<media>::Packets(const Packets<media>& other)
    : id(other.id),
      src(other.src),
      pkts(other.pkts),
      time_base(other.time_base),
      timestamp(other.timestamp),
      codec(other.codec) {}

template <MediaType media>
Packets<media>& Packets<media>::operator=(const Packets<media>& other) {
  Packets<media> tmp{other};
  *this = std::move(tmp);
  return *this;
}

template <MediaType media>
Packets<media>::Packets(Packets<media>&& other) noexcept {
  *this = std::move(other);
}

template <MediaType media>
Packets<media>& Packets<media>::operator=(Packets<media>&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(src, other.src);
  swap(pkts, other.pkts);
  swap(time_base, other.time_base);
  swap(timestamp, other.timestamp);
  swap(codec, other.codec);
  return *this;
}

const std::vector<AVPacket*>& PacketSeries::get_packets() const {
  return container;
}

Generator<RawPacketData> PacketSeries::iter_data() const {
  for (auto& pkt : container) {
    co_yield RawPacketData{pkt->data, pkt->size, pkt->pts};
  }
}

template struct Packets<MediaType::Audio>;
template struct Packets<MediaType::Video>;
template struct Packets<MediaType::Image>;

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
  auto& src_packets = src->pkts.get_packets();
  auto ret = std::make_unique<VideoPackets>();
  ret->src = src->src;
  ret->codec = src->codec;
  ret->timestamp = src->timestamp;
  for (size_t t = start; t < end; ++t) {
    ret->pkts.push(CHECK_AVALLOCATE(av_packet_clone(src_packets[t])));
  }
  return ret;
}

} // namespace

std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    std::vector<size_t> indices) {
  auto& src_packets = src->pkts.get_packets();
  // If timestamp is set, then there are frames before the window.
  // `indices` are supposed to be within the window.
  // So we adjust the `indices` by shifitng the number of frames before the
  // window.
  if (src->timestamp) {
    auto [start, end] = *(src->timestamp);
    size_t offset = 0;
    auto tb = src->time_base;
    for (auto& packet : src_packets) {
      auto pts = static_cast<double>(packet->pts) * tb.num / tb.den;
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

template <MediaType media>
std::optional<std::tuple<double, double>> get_pts(
    const Packets<media>& packets) {
  const auto& pkts = packets.pkts.get_packets();
  if (!pkts.size()) {
    return std::nullopt;
  }
  double t0 = pkts[0]->pts, tN = pkts[pkts.size() - 1]->pts;
  auto tb = packets.time_base;
  return std::tuple<double, double>{t0 * tb.num / tb.den, tN * tb.num / tb.den};
};

template std::optional<std::tuple<double, double>> get_pts(
    const Packets<MediaType::Audio>& packets);
template std::optional<std::tuple<double, double>> get_pts(
    const Packets<MediaType::Video>& packets);

} // namespace spdl::core
