/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include <libspdl/core/rational_utils.h>

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/rational.h>
}
namespace spdl::core {

PacketSeries::PacketSeries() {}

PacketSeries::PacketSeries(const PacketSeries& other) {
  for (const AVPacket* pkt : other.container_) {
    container_.push_back(CHECK_AVALLOCATE(av_packet_clone(pkt)));
  }
}

PacketSeries::PacketSeries(PacketSeries&& other) noexcept {
  *this = std::move(other);
}

PacketSeries& PacketSeries::operator=(const PacketSeries& other) {
  PacketSeries tmp(other);
  *this = std::move(tmp);
  return *this;
}

PacketSeries& PacketSeries::operator=(PacketSeries&& other) noexcept {
  using std::swap;
  swap(container_, other.container_);
  return *this;
}

PacketSeries::~PacketSeries() {
  std::for_each(container_.begin(), container_.end(), [](AVPacket* p) {
    if (p) {
      av_packet_unref(p);
      av_packet_free(&p);
    }
  });
}

void PacketSeries::push(AVPacket* p) {
  if (!p) {
    SPDL_FAIL_INTERNAL("Packet is NULL.");
  }
  container_.push_back(p);
}

template <MediaType media>
Packets<media>::Packets(
    const std::string& src_,
    int index,
    Codec<media>&& codec_,
    const std::optional<TimeWindow>& timestamp_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      stream_index(index),
      time_base(codec_.get_time_base()),
      timestamp(timestamp_),
      codec(std::move(codec_)) {
  TRACE_EVENT(
      "decoding", "Packets::Packets", perfetto::Flow::ProcessScoped(id));
}

template <MediaType media>
Packets<media>::Packets(
    const std::string& src_,
    int index,
    const Rational& time_base_,
    const std::optional<TimeWindow>& timestamp_)
    : id(reinterpret_cast<uintptr_t>(this)),
      src(src_),
      stream_index(index),
      time_base(time_base_),
      timestamp(timestamp_),
      codec(std::nullopt) {
  TRACE_EVENT(
      "decoding", "Packets::Packets", perfetto::Flow::ProcessScoped(id));
}
template <MediaType media>
Packets<media>::Packets(uintptr_t id_, int index, Rational time_base_)
    : id(id_),
      src(fmt::format("{}", id_)),
      stream_index(index),
      time_base(time_base_),
      timestamp(std::nullopt),
      codec(std::nullopt) {
  TRACE_EVENT(
      "decoding", "Packets::Packets", perfetto::Flow::ProcessScoped(id));
}

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
  swap(is_view, other.is_view);
  return *this;
}

const std::vector<AVPacket*>& PacketSeries::get_packets() const {
  return container_;
}

Generator<RawPacketData> PacketSeries::iter_data() const {
  for (auto& pkt : container_) {
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
  // be discarded.
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
  keyframe_pts.emplace_back(0, packets[0]->pts);
  for (size_t i = 1; i < packets.size(); ++i) {
    auto pkt = packets[i];
    if (pkt->flags & AV_PKT_FLAG_KEY) {
      keyframe_pts.emplace_back(i, pkt->pts);
    }
  }
  keyframe_pts.emplace_back(packets.size(), LLONG_MAX);

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
    ret.emplace_back(start, end, num_invalid);
  }
  return ret;
}

VideoPacketsPtr
extract_packets(const VideoPacketsPtr& src, size_t start, size_t end) {
  auto& src_packets = src->pkts.get_packets();
  auto ret = std::make_unique<VideoPackets>();
  ret->src = src->src;
  ret->codec = src->codec;
  // Do not preserve timestamp as indices are already adjusted
  ret->timestamp = std::nullopt;
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
  // So we adjust the `indices` by shifting the number of frames before the
  // window.
  if (src->timestamp) {
    auto start = std::get<0>(*(src->timestamp));
    size_t offset = 0;
    auto tb = src->time_base;
    for (auto& packet : src_packets) {
      auto pts = to_rational(packet->pts, tb);
      if (av_cmp_q(pts, start) < 0) {
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
      ret.emplace_back(extract_packets(src, start, end), indices_in_window);
    }
    if (i >= indices.size()) {
      break;
    }
  }
  return ret;
}

template <MediaType media>
std::vector<double> get_timestamps(const Packets<media>& packets, bool raw) {
  const auto& pkts = packets.pkts.get_packets();

  std::vector<double> ret{};
  ret.reserve(pkts.size());

  // When timestamp is not set, include all packets
  if (!packets.timestamp || raw) {
    for (const auto& pkt : pkts) {
      AVRational pts = to_rational(pkt->pts, packets.time_base);
      ret.emplace_back(av_q2d(pts));
    }
  } else {
    // When timestamp is set, filter packets within the window
    auto [s, e] = *(packets.timestamp);
    for (const auto& pkt : pkts) {
      AVRational pts = to_rational(pkt->pts, packets.time_base);
      if (is_within_window(pts, s, e)) {
        ret.emplace_back(av_q2d(pts));
      }
    }
  }

  if (!raw) {
    std::sort(ret.begin(), ret.end());
  }
  return ret;
}

template std::vector<double> get_timestamps(const AudioPackets&, bool);
template std::vector<double> get_timestamps(const VideoPackets&, bool);
template std::vector<double> get_timestamps(const ImagePackets&, bool);

////////////////////////////////////////////////////////////////////////////////
// Serialization helpers
////////////////////////////////////////////////////////////////////////////////

namespace {

constexpr uint32_t SERIALIZATION_MAGIC = 0x53504B54; // "SPKT"
constexpr uint8_t SERIALIZATION_VERSION = 1;

// Each serialized packet payload is followed by this many zeroed bytes. FFmpeg
// decoders may over-read up to AV_INPUT_BUFFER_PADDING_SIZE past the end of
// packet data, so the padding lets a payload be restored as a zero-copy view
// (deserialize_packets_view) without copying it into an av_malloc'd buffer.
constexpr size_t SERIALIZATION_PADDING = AV_INPUT_BUFFER_PADDING_SIZE;

// Each payload is aligned to this many bytes *within the serialized blob*. When
// the arena writer also places the blob on an aligned boundary, a zero-copy
// view lands on an aligned address (some SIMD-optimized decoders need this). 64
// bytes matches av_malloc and the widest SIMD. Keep in sync with the arena
// writer's alignment (spdl/pipeline/_arena/_pool.py).
constexpr size_t SERIALIZATION_ALIGNMENT = 64;

class ByteWriter {
  std::vector<uint8_t>& buf_;

 public:
  explicit ByteWriter(std::vector<uint8_t>& buf) : buf_(buf) {}

  void write_raw(const void* data, size_t size) {
    auto* p = reinterpret_cast<const uint8_t*>(data);
    buf_.insert(buf_.end(), p, p + size);
  }

  template <typename T>
  void write(T val) {
    write_raw(&val, sizeof(T));
  }

  void write_bytes(const uint8_t* data, size_t size) {
    if (!data) {
      write<int32_t>(0);
      return;
    }
    write<int32_t>(static_cast<int32_t>(size));
    if (size > 0) {
      write_raw(data, size);
    }
  }

  void write_string(const std::string& s) {
    write<int32_t>(static_cast<int32_t>(s.size()));
    if (!s.empty()) {
      write_raw(s.data(), s.size());
    }
  }

  // Pad the buffer with zeros up to the next SERIALIZATION_ALIGNMENT boundary
  // (offsets are relative to the blob start).
  void pad_to_alignment() {
    if (auto rem = buf_.size() % SERIALIZATION_ALIGNMENT; rem != 0) {
      buf_.insert(buf_.end(), SERIALIZATION_ALIGNMENT - rem, 0);
    }
  }

  // Write a packet payload: size, alignment padding, raw bytes, then
  // SERIALIZATION_PADDING zeroed bytes. The alignment padding places the
  // payload on an aligned blob offset, and the trailing padding lets it be
  // restored as a zero-copy view (decoders over-read the trailing padding).
  void write_payload(const uint8_t* data, int size) {
    if (!data || size <= 0) {
      write<int32_t>(0);
      return;
    }
    write<int32_t>(size);
    pad_to_alignment();
    write_raw(data, size);
    buf_.insert(buf_.end(), SERIALIZATION_PADDING, 0);
  }
};

class ByteReader {
  const uint8_t* data_;
  size_t size_;
  size_t offset_ = 0;

 public:
  ByteReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  void read_raw(void* dst, size_t size) {
    if (offset_ + size > size_) {
      throw std::runtime_error("Deserialization buffer underflow");
    }
    std::memcpy(dst, data_ + offset_, size);
    offset_ += size;
  }

  template <typename T>
  T read() {
    T val;
    read_raw(&val, sizeof(T));
    return val;
  }

  std::vector<uint8_t> read_bytes() {
    auto size = read<int32_t>();
    if (size < 0) {
      throw std::runtime_error("Deserialization error: negative size");
    }
    std::vector<uint8_t> result(size);
    if (size > 0) {
      read_raw(result.data(), size);
    }
    return result;
  }

  std::string read_string() {
    auto size = read<int32_t>();
    if (size < 0) {
      throw std::runtime_error("Deserialization error: negative size");
    }
    std::string result(size, '\0');
    if (size > 0) {
      read_raw(result.data(), size);
    }
    return result;
  }

  // Pointer to the current read position; valid while the buffer is alive.
  const uint8_t* peek() const {
    return data_ + offset_;
  }

  void skip(size_t size) {
    if (offset_ + size > size_) {
      throw std::runtime_error("Deserialization buffer underflow");
    }
    offset_ += size;
  }

  // Advance to the next SERIALIZATION_ALIGNMENT boundary (mirrors
  // ByteWriter::pad_to_alignment).
  void align_to_alignment() {
    if (auto rem = offset_ % SERIALIZATION_ALIGNMENT; rem != 0) {
      skip(SERIALIZATION_ALIGNMENT - rem);
    }
  }

  // Copy out a payload written by ByteWriter::write_payload (consumes the
  // alignment padding and the trailing padding).
  std::vector<uint8_t> read_payload() {
    auto size = read<int32_t>();
    if (size < 0) {
      throw std::runtime_error("Deserialization error: negative size");
    }
    std::vector<uint8_t> result(size);
    if (size > 0) {
      align_to_alignment();
      read_raw(result.data(), size);
      skip(SERIALIZATION_PADDING);
    }
    return result;
  }

  // Return a pointer to a payload in place and advance past it (and its
  // padding) without copying. The pointer sits on an aligned blob offset and is
  // valid while the buffer is alive.
  std::pair<const uint8_t*, int> view_payload() {
    auto size = read<int32_t>();
    if (size < 0) {
      throw std::runtime_error("Deserialization error: negative size");
    }
    if (size == 0) {
      return {nullptr, 0};
    }
    align_to_alignment();
    const uint8_t* p = peek();
    skip(static_cast<size_t>(size) + SERIALIZATION_PADDING);
    return {p, size};
  }
};

void serialize_packet(ByteWriter& w, const AVPacket* pkt) {
#if LIBAVCODEC_VERSION_MAJOR >= 59
  if (pkt->opaque) {
    throw std::runtime_error(
        "Cannot serialize AVPacket with non-NULL opaque pointer");
  }
  if (pkt->opaque_ref) {
    throw std::runtime_error(
        "Cannot serialize AVPacket with non-NULL opaque_ref");
  }
#endif

  w.write<int64_t>(pkt->pts);
  w.write<int64_t>(pkt->dts);
  w.write<int32_t>(pkt->flags);
  w.write<int64_t>(pkt->duration);
  w.write_payload(pkt->data, pkt->size);

  w.write<int32_t>(pkt->side_data_elems);
  for (int i = 0; i < pkt->side_data_elems; ++i) {
    w.write<int32_t>(static_cast<int32_t>(pkt->side_data[i].type));
    w.write_bytes(pkt->side_data[i].data, pkt->side_data[i].size);
  }
}

AVPacket* deserialize_packet(ByteReader& r) {
  detail::AVPacketPtr pkt(av_packet_alloc());
  if (!pkt) {
    throw std::runtime_error("Failed to allocate AVPacket");
  }

  auto pts = r.read<int64_t>();
  auto dts = r.read<int64_t>();
  auto flags = r.read<int32_t>();
  auto duration = r.read<int64_t>();

  auto data = r.read_payload();
  if (!data.empty()) {
    if (av_new_packet(pkt.get(), static_cast<int>(data.size())) < 0) {
      throw std::runtime_error("Failed to allocate packet data");
    }
    std::memcpy(pkt->data, data.data(), data.size());
  }

  pkt->pts = pts;
  pkt->dts = dts;
  pkt->flags = flags;
  pkt->duration = duration;

  auto num_side_data = r.read<int32_t>();
  for (int32_t i = 0; i < num_side_data; ++i) {
    auto type = static_cast<AVPacketSideDataType>(r.read<int32_t>());
    auto sd_data = r.read_bytes();
    auto* sd = av_packet_new_side_data(
        pkt.get(), type, static_cast<int>(sd_data.size()));
    if (!sd) {
      throw std::runtime_error("Failed to allocate packet side data");
    }
    std::memcpy(sd, sd_data.data(), sd_data.size());
  }

  return pkt.release();
}

// Like deserialize_packet, but the payload is a non-owning view into the
// reader's buffer: pkt->data points into it and pkt->buf stays NULL. Side data
// is still copied (owned). The caller must keep the buffer alive for the
// packet's life.
AVPacket* deserialize_view_packet(ByteReader& r) {
  detail::AVPacketPtr pkt(av_packet_alloc());
  if (!pkt) {
    throw std::runtime_error("Failed to allocate AVPacket");
  }

  auto pts = r.read<int64_t>();
  auto dts = r.read<int64_t>();
  auto flags = r.read<int32_t>();
  auto duration = r.read<int64_t>();

  auto [data_ptr, size] = r.view_payload();
  if (size > 0) {
    // Non-owning view: point at the buffer and leave buf == NULL so FFmpeg
    // treats the data as unreferenced (clones/refs deep-copy).
    pkt->data = const_cast<uint8_t*>(data_ptr);
    pkt->size = size;
  }

  pkt->pts = pts;
  pkt->dts = dts;
  pkt->flags = flags;
  pkt->duration = duration;

  auto num_side_data = r.read<int32_t>();
  for (int32_t i = 0; i < num_side_data; ++i) {
    auto type = static_cast<AVPacketSideDataType>(r.read<int32_t>());
    auto sd_data = r.read_bytes();
    auto* sd = av_packet_new_side_data(
        pkt.get(), type, static_cast<int>(sd_data.size()));
    if (!sd) {
      throw std::runtime_error("Failed to allocate packet side data");
    }
    std::memcpy(sd, sd_data.data(), sd_data.size());
  }

  return pkt.release();
}

void serialize_codec_parameters(ByteWriter& w, const AVCodecParameters* par) {
  w.write<int32_t>(static_cast<int32_t>(par->codec_type));
  w.write<int32_t>(static_cast<int32_t>(par->codec_id));
  w.write<uint32_t>(par->codec_tag);
  w.write<int32_t>(par->format);
  w.write<int64_t>(par->bit_rate);
  w.write<int32_t>(par->bits_per_coded_sample);
  w.write<int32_t>(par->bits_per_raw_sample);
  w.write<int32_t>(par->profile);
  w.write<int32_t>(par->level);
  w.write<int32_t>(par->width);
  w.write<int32_t>(par->height);
  w.write<int32_t>(par->sample_aspect_ratio.num);
  w.write<int32_t>(par->sample_aspect_ratio.den);
#if LIBAVCODEC_VERSION_MAJOR >= 60
  w.write<int32_t>(par->framerate.num);
  w.write<int32_t>(par->framerate.den);
#else
  w.write<int32_t>(0);
  w.write<int32_t>(1);
#endif
  w.write<int32_t>(par->field_order);
  w.write<int32_t>(par->color_range);
  w.write<int32_t>(par->color_primaries);
  w.write<int32_t>(par->color_trc);
  w.write<int32_t>(par->color_space);
  w.write<int32_t>(par->chroma_location);
  w.write<int32_t>(par->video_delay);
  w.write<int32_t>(par->sample_rate);
  w.write<int32_t>(par->block_align);
  w.write<int32_t>(par->frame_size);
  w.write<int32_t>(par->initial_padding);
  w.write<int32_t>(par->trailing_padding);
  w.write<int32_t>(par->seek_preroll);

  // Channel layout
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 17, 100)
  w.write<int32_t>(static_cast<int32_t>(par->ch_layout.order));
  w.write<int32_t>(par->ch_layout.nb_channels);
  if (par->ch_layout.order == AV_CHANNEL_ORDER_NATIVE ||
      par->ch_layout.order == AV_CHANNEL_ORDER_AMBISONIC) {
    w.write<uint64_t>(par->ch_layout.u.mask);
  } else if (par->ch_layout.order == AV_CHANNEL_ORDER_CUSTOM) {
    for (int i = 0; i < par->ch_layout.nb_channels; ++i) {
      if (par->ch_layout.u.map[i].opaque) {
        throw std::runtime_error(
            "Cannot serialize AVChannelCustom with non-NULL opaque pointer");
      }
      w.write<int32_t>(static_cast<int32_t>(par->ch_layout.u.map[i].id));
      w.write_raw(par->ch_layout.u.map[i].name, 16);
    }
  }
#else
  w.write<int32_t>(par->channels);
  w.write<uint64_t>(par->channel_layout);
#endif

  // Extradata
  w.write_bytes(par->extradata, par->extradata_size);

  // Coded side data (available in FFmpeg 6.1+ / libavcodec 60.31+)
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(60, 31, 102)
  w.write<int32_t>(par->nb_coded_side_data);
  for (int i = 0; i < par->nb_coded_side_data; ++i) {
    w.write<int32_t>(static_cast<int32_t>(par->coded_side_data[i].type));
    w.write_bytes(par->coded_side_data[i].data, par->coded_side_data[i].size);
  }
#else
  w.write<int32_t>(0);
#endif
}

AVCodecParameters* deserialize_codec_parameters(ByteReader& r) {
  detail::AVCodecParametersPtr par(avcodec_parameters_alloc());
  if (!par) {
    throw std::runtime_error("Failed to allocate AVCodecParameters");
  }

  par->codec_type = static_cast<AVMediaType>(r.read<int32_t>());
  par->codec_id = static_cast<AVCodecID>(r.read<int32_t>());
  par->codec_tag = r.read<uint32_t>();
  par->format = r.read<int32_t>();
  par->bit_rate = r.read<int64_t>();
  par->bits_per_coded_sample = r.read<int32_t>();
  par->bits_per_raw_sample = r.read<int32_t>();
  par->profile = r.read<int32_t>();
  par->level = r.read<int32_t>();
  par->width = r.read<int32_t>();
  par->height = r.read<int32_t>();
  par->sample_aspect_ratio.num = r.read<int32_t>();
  par->sample_aspect_ratio.den = r.read<int32_t>();
#if LIBAVCODEC_VERSION_MAJOR >= 60
  par->framerate.num = r.read<int32_t>();
  par->framerate.den = r.read<int32_t>();
#else
  r.read<int32_t>(); // framerate.num (not available in this version)
  r.read<int32_t>(); // framerate.den
#endif
  par->field_order = static_cast<AVFieldOrder>(r.read<int32_t>());
  par->color_range = static_cast<AVColorRange>(r.read<int32_t>());
  par->color_primaries = static_cast<AVColorPrimaries>(r.read<int32_t>());
  par->color_trc =
      static_cast<AVColorTransferCharacteristic>(r.read<int32_t>());
  par->color_space = static_cast<AVColorSpace>(r.read<int32_t>());
  par->chroma_location = static_cast<AVChromaLocation>(r.read<int32_t>());
  par->video_delay = r.read<int32_t>();
  par->sample_rate = r.read<int32_t>();
  par->block_align = r.read<int32_t>();
  par->frame_size = r.read<int32_t>();
  par->initial_padding = r.read<int32_t>();
  par->trailing_padding = r.read<int32_t>();
  par->seek_preroll = r.read<int32_t>();

  // Channel layout
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 17, 100)
  par->ch_layout.order = static_cast<AVChannelOrder>(r.read<int32_t>());
  par->ch_layout.nb_channels = r.read<int32_t>();
  if (par->ch_layout.order == AV_CHANNEL_ORDER_NATIVE ||
      par->ch_layout.order == AV_CHANNEL_ORDER_AMBISONIC) {
    par->ch_layout.u.mask = r.read<uint64_t>();
  } else if (par->ch_layout.order == AV_CHANNEL_ORDER_CUSTOM) {
    par->ch_layout.u.map = static_cast<AVChannelCustom*>(
        av_calloc(par->ch_layout.nb_channels, sizeof(AVChannelCustom)));
    if (!par->ch_layout.u.map) {
      throw std::runtime_error("Failed to allocate channel map");
    }
    for (int i = 0; i < par->ch_layout.nb_channels; ++i) {
      par->ch_layout.u.map[i].id = static_cast<AVChannel>(r.read<int32_t>());
      r.read_raw(par->ch_layout.u.map[i].name, 16);
      par->ch_layout.u.map[i].opaque = nullptr;
    }
  }
#else
  par->channels = r.read<int32_t>();
  par->channel_layout = r.read<uint64_t>();
#endif

  // Extradata
  auto extradata = r.read_bytes();
  if (!extradata.empty()) {
    par->extradata = static_cast<uint8_t*>(
        av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!par->extradata) {
      throw std::runtime_error("Failed to allocate extradata");
    }
    std::memcpy(par->extradata, extradata.data(), extradata.size());
    par->extradata_size = static_cast<int>(extradata.size());
  }

  // Coded side data (available in FFmpeg 6.1+ / libavcodec 60.31+)
  auto nb_coded_side_data = r.read<int32_t>();
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(60, 31, 102)
  par->nb_coded_side_data = nb_coded_side_data;
  if (nb_coded_side_data > 0) {
    par->coded_side_data = static_cast<AVPacketSideData*>(
        av_calloc(nb_coded_side_data, sizeof(AVPacketSideData)));
    if (!par->coded_side_data) {
      throw std::runtime_error("Failed to allocate coded side data");
    }
    for (int32_t i = 0; i < nb_coded_side_data; ++i) {
      par->coded_side_data[i].type =
          static_cast<AVPacketSideDataType>(r.read<int32_t>());
      auto sd_data = r.read_bytes();
      par->coded_side_data[i].size = sd_data.size();
      par->coded_side_data[i].data =
          static_cast<uint8_t*>(av_malloc(sd_data.size()));
      if (!par->coded_side_data[i].data) {
        throw std::runtime_error("Failed to allocate coded side data payload");
      }
      std::memcpy(par->coded_side_data[i].data, sd_data.data(), sd_data.size());
    }
  }
#else
  // Older FFmpeg versions don't have coded_side_data on AVCodecParameters.
  // Just consume the data from the stream.
  for (int32_t i = 0; i < nb_coded_side_data; ++i) {
    r.read<int32_t>(); // type
    r.read_bytes(); // data
  }
#endif

  return par.release();
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Serialization implementation
////////////////////////////////////////////////////////////////////////////////

template <MediaType media>
std::vector<uint8_t> serialize_packets(const Packets<media>& packets) {
  std::vector<uint8_t> buf;
  ByteWriter w(buf);

  w.write<uint32_t>(SERIALIZATION_MAGIC);
  w.write<uint8_t>(SERIALIZATION_VERSION);
  w.write<uint8_t>(static_cast<uint8_t>(media));

  w.write_string(packets.src);
  w.write<int32_t>(packets.stream_index);
  w.write<int32_t>(packets.time_base.num);
  w.write<int32_t>(packets.time_base.den);

  // Timestamp
  w.write<uint8_t>(packets.timestamp.has_value() ? 1 : 0);
  if (packets.timestamp) {
    auto [start, end] = *packets.timestamp;
    w.write<int32_t>(start.num);
    w.write<int32_t>(start.den);
    w.write<int32_t>(end.num);
    w.write<int32_t>(end.den);
  }

  // Codec
  w.write<uint8_t>(packets.codec.has_value() ? 1 : 0);
  if (packets.codec) {
    auto tb = packets.codec->get_time_base();
    auto fr = packets.codec->get_frame_rate();
    w.write<int32_t>(tb.num);
    w.write<int32_t>(tb.den);
    w.write<int32_t>(fr.num);
    w.write<int32_t>(fr.den);
    serialize_codec_parameters(w, packets.codec->get_parameters());
  }

  // Packets
  const auto& pkts = packets.pkts.get_packets();
  w.write<int32_t>(static_cast<int32_t>(pkts.size()));
  for (const auto* pkt : pkts) {
    serialize_packet(w, pkt);
  }

  return buf;
}

// Read everything up to (but not including) the packet count: the header,
// src/stream_index/time_base, timestamp and codec. Shared by the copy and view
// deserializers; leaves `r` positioned at the int32 packet count.
template <MediaType media>
static std::unique_ptr<Packets<media>> read_packets_header(ByteReader& r) {
  auto magic = r.read<uint32_t>();
  if (magic != SERIALIZATION_MAGIC) {
    throw std::runtime_error("Invalid serialization magic number");
  }
  auto version = r.read<uint8_t>();
  if (version != SERIALIZATION_VERSION) {
    throw std::runtime_error(
        fmt::format(
            "Unsupported serialization version: {}",
            static_cast<int>(version)));
  }
  auto media_type = static_cast<MediaType>(r.read<uint8_t>());
  if (media_type != media) {
    throw std::runtime_error("Media type mismatch during deserialization");
  }

  auto result = std::make_unique<Packets<media>>();
  result->id = 0;
  result->src = r.read_string();
  result->stream_index = r.read<int32_t>();
  result->time_base.num = r.read<int32_t>();
  result->time_base.den = r.read<int32_t>();

  // Timestamp
  auto has_timestamp = r.read<uint8_t>();
  if (has_timestamp) {
    Rational start, end;
    start.num = r.read<int32_t>();
    start.den = r.read<int32_t>();
    end.num = r.read<int32_t>();
    end.den = r.read<int32_t>();
    result->timestamp = TimeWindow{start, end};
  }

  // Codec
  auto has_codec = r.read<uint8_t>();
  if (has_codec) {
    Rational tb, fr;
    tb.num = r.read<int32_t>();
    tb.den = r.read<int32_t>();
    fr.num = r.read<int32_t>();
    fr.den = r.read<int32_t>();
    auto* par = deserialize_codec_parameters(r);
    result->codec.emplace(Codec<media>(par, tb, fr));
    avcodec_parameters_free(&par);
  }

  return result;
}

template <MediaType media>
std::unique_ptr<Packets<media>> deserialize_packets(
    const std::vector<uint8_t>& data) {
  ByteReader r(data.data(), data.size());
  auto result = read_packets_header<media>(r);

  auto num_packets = r.read<int32_t>();
  for (int32_t i = 0; i < num_packets; ++i) {
    result->pkts.push(deserialize_packet(r));
  }

  return result;
}

template <MediaType media>
std::unique_ptr<Packets<media>> deserialize_packets_view(
    const uint8_t* data,
    size_t size) {
  ByteReader r(data, size);
  auto result = read_packets_header<media>(r);

  auto num_packets = r.read<int32_t>();
  for (int32_t i = 0; i < num_packets; ++i) {
    result->pkts.push(deserialize_view_packet(r));
  }
  result->is_view = true;

  return result;
}

template std::vector<uint8_t> serialize_packets(const AudioPackets&);
template std::vector<uint8_t> serialize_packets(const VideoPackets&);
template std::vector<uint8_t> serialize_packets(const ImagePackets&);

template std::unique_ptr<AudioPackets> deserialize_packets<MediaType::Audio>(
    const std::vector<uint8_t>&);
template std::unique_ptr<VideoPackets> deserialize_packets<MediaType::Video>(
    const std::vector<uint8_t>&);
template std::unique_ptr<ImagePackets> deserialize_packets<MediaType::Image>(
    const std::vector<uint8_t>&);

template std::unique_ptr<AudioPackets>
deserialize_packets_view<MediaType::Audio>(const uint8_t*, size_t);
template std::unique_ptr<VideoPackets>
deserialize_packets_view<MediaType::Video>(const uint8_t*, size_t);
template std::unique_ptr<ImagePackets>
deserialize_packets_view<MediaType::Image>(const uint8_t*, size_t);

} // namespace spdl::core
