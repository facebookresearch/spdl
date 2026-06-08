/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/codec.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

struct AVPacket;

namespace spdl::core {

/// Media packets template forward declaration.
template <MediaType media>
struct Packets;

/// Audio packets type alias.
using AudioPackets = Packets<MediaType::Audio>;
/// Video packets type alias.
using VideoPackets = Packets<MediaType::Video>;
/// Image packets type alias.
using ImagePackets = Packets<MediaType::Image>;

/// Unique pointer to Packets.
template <MediaType media>
using PacketsPtr = std::unique_ptr<Packets<media>>;

/// Unique pointer to AudioPackets.
using AudioPacketsPtr = PacketsPtr<MediaType::Audio>;
/// Unique pointer to VideoPackets.
using VideoPacketsPtr = PacketsPtr<MediaType::Video>;
/// Unique pointer to ImagePackets.
using ImagePacketsPtr = PacketsPtr<MediaType::Image>;

/// Variant type holding any packets type.
using AnyPackets =
    std::variant<AudioPacketsPtr, VideoPacketsPtr, ImagePacketsPtr>;

class PacketSeries;
/// Unique pointer to PacketSeries.
using PacketSeriesPtr = std::unique_ptr<PacketSeries>;

/// Raw packet data without AVPacket dependency.
///
/// Exposes packet data without including FFmpeg headers, allowing
/// NVDEC decoder to operate without libavutil dependencies.
struct RawPacketData {
  /// Pointer to packet data.
  uint8_t* data;
  /// Size of packet data in bytes.
  int size;
  /// Presentation timestamp.
  int64_t pts;
};

/// Container managing a series of compressed packets.
///
/// PacketSeries manages the lifetime of FFmpeg AVPacket pointers,
/// providing safe access to compressed media data.
class PacketSeries {
  friend struct Packets<MediaType::Audio>;
  friend struct Packets<MediaType::Video>;
  friend struct Packets<MediaType::Image>;

  std::vector<AVPacket*> container_ = {};

 public:
  /// Default constructor.
  PacketSeries();

  /// Destructor releases AVPacket resources.
  ~PacketSeries();

  /// Copy constructor.
  explicit PacketSeries(const PacketSeries&);

  /// Copy assignment operator.
  PacketSeries& operator=(const PacketSeries&);

  /// Move constructor.
  PacketSeries(PacketSeries&& other) noexcept;

  /// Move assignment operator.
  PacketSeries& operator=(PacketSeries&& other) noexcept;

  /// Add a new packet to the series.
  ///
  /// @param packet AVPacket pointer to add.
  void push(AVPacket* packet);

  /// Get all packets in the series.
  ///
  /// @return Vector of AVPacket pointers.
  const std::vector<AVPacket*>& get_packets() const;

  /// Iterate through packet data.
  ///
  /// @return Generator yielding RawPacketData for each packet.
  Generator<RawPacketData> iter_data() const;
};

/// Compressed media packets with metadata.
///
/// Packets structure carries compressed packets from demuxer to decoder,
/// along with codec information and timing metadata. Similar to Frames,
/// AVPacket pointers are managed for bulk resource release.
///
/// @tparam media Media type (Audio, Video, or Image).
template <MediaType media>
struct Packets {
  /// Trace ID for debugging.
  uintptr_t id{};
  /// Source URI or identifier.
  std::string src;
  /// Stream index in the source.
  int stream_index;

  /// Series of compressed packets.
  PacketSeries pkts;
  /// Time base for packet timestamps.
  Rational time_base{};
  /// Optional time window specified by user.
  std::optional<TimeWindow> timestamp;

  /// Codec information for decoding.
  /// Available when packets are demuxed by demux_window method.
  /// For streaming demuxing, codec must be fetched from demuxer directly.
  std::optional<Codec<media>> codec;

  /// When true, the AVPacket payloads are non-owning *views* into external
  /// memory (e.g. a shared-memory segment), built by
  /// ``deserialize_packets_view``. The caller guarantees that backing memory
  /// outlives this object — analogous to ``std::string_view``. Cloning or
  /// re-serializing a view materializes an owning copy (FFmpeg copies when
  /// ``AVPacket::buf == NULL``), so derived objects are safe to outlive the
  /// backing memory.
  bool is_view{false};

  /// Default constructor.
  Packets() = default;

  /// Construct packets from demuxer for one-time decoding.
  ///
  /// Includes codec information needed to initialize the decoder.
  ///
  /// @param src Source URI.
  /// @param index Stream index.
  /// @param codec Codec information.
  /// @param timestamp Optional time window user specified.
  Packets(
      const std::string& src,
      int index,
      Codec<media>&& codec,
      const std::optional<TimeWindow>& timestamp = {});

  /// Construct packets from demuxer for streaming.
  ///
  /// Decoder is initialized separately, so codec is not needed here.
  ///
  /// @param src Source URI.
  /// @param index Stream index.
  /// @param time_base Time base for timestamps.
  /// @param timestamp Optional time window user specified.
  Packets(
      const std::string& src,
      int index,
      const Rational& time_base,
      const std::optional<TimeWindow>& timestamp = {});

  /// Construct packets from encoder for muxing.
  ///
  /// @param id Trace ID.
  /// @param stream_index Output stream index.
  /// @param time_base Time base for timestamps.
  Packets(uintptr_t id, int stream_index, Rational time_base);

  /// Copy constructor.
  explicit Packets(const Packets<media>&);

  /// Copy assignment operator.
  Packets<media>& operator=(const Packets<media>&);

  /// Move constructor.
  Packets(Packets<media>&&) noexcept;

  /// Move assignment operator.
  Packets<media>& operator=(Packets<media>&&) noexcept;

  /// Destructor.
  ~Packets() = default;
};

/// Extract packets at specific indices from video packets.
///
/// @param src Source video packets.
/// @param indices Indices of packets to extract.
/// @return Vector of tuples containing packets and their original indices.
std::vector<std::tuple<VideoPacketsPtr, std::vector<size_t>>>
extract_packets_at_indices(
    const VideoPacketsPtr& src,
    std::vector<size_t> indices);

/// Get packet timestamps in seconds.
///
/// By default, applies user-specified window filtering and sorts timestamps
/// for video packets. Setting raw=true disables filtering and sorting,
/// returning timestamps as-is.
///
/// @tparam media Media type (Audio, Video, or Image).
/// @param packets Packets to extract timestamps from.
/// @param raw If true, disable filtering and sorting.
/// @return Vector of timestamps in seconds.
template <MediaType media>
std::vector<double> get_timestamps(
    const Packets<media>& packets,
    bool raw = false);

////////////////////////////////////////////////////////////////////////////////
// Serialization
////////////////////////////////////////////////////////////////////////////////

/// Serialize Packets to a byte vector for IPC (e.g., multiprocessing).
///
/// NOTE: This byte format is **internal and process-local**. It is only ever
/// produced and consumed by the same build (e.g. an `iterate_in_subprocess`
/// worker and its parent), and is **not** stable for on-disk persistence or
/// cross-version exchange. The layout may change without a version bump; the
/// embedded magic/version bytes are an internal sanity check, not a
/// compatibility contract. Each packet payload is written followed by
/// `AV_INPUT_BUFFER_PADDING_SIZE` zeroed bytes so it can be restored as a
/// zero-copy view (see `deserialize_packets_view`).
///
/// Throws std::runtime_error if any non-serializable pointer field
/// (AVPacket::opaque, AVPacket::opaque_ref) is non-NULL.
///
/// @tparam media Media type (Audio, Video, or Image).
/// @param packets Packets to serialize.
/// @return Serialized byte vector.
template <MediaType media>
std::vector<uint8_t> serialize_packets(const Packets<media>& packets);

/// Deserialize a byte buffer (produced by `serialize_packets`) back to Packets,
/// copying each payload into a freshly owned AVPacket buffer.
///
/// See `serialize_packets` for the note that the format is
/// internal/process-local.
///
/// @tparam media Media type (Audio, Video, or Image).
/// @param data Serialized byte vector.
/// @return Deserialized Packets (owning their payloads).
template <MediaType media>
std::unique_ptr<Packets<media>> deserialize_packets(
    const std::vector<uint8_t>& data);

/// Deserialize a byte buffer **without copying payloads**: each AVPacket points
/// directly into `data` (`AVPacket::buf == NULL`, non-owning). The returned
/// Packets has `is_view == true`.
///
/// The caller MUST keep `data` alive and unmodified for the entire lifetime of
/// the returned Packets (and of any non-clone references taken from it). The
/// buffer must have been produced by `serialize_packets` (which writes the
/// required trailing payload padding). Cloning/re-referencing a view packet
/// deep-copies (FFmpeg copies when `buf == NULL`), so derived packets are safe.
///
/// @tparam media Media type (Audio, Video, or Image).
/// @param data Pointer to the serialized buffer.
/// @param size Size of the serialized buffer in bytes.
/// @return Deserialized Packets viewing `data`.
template <MediaType media>
std::unique_ptr<Packets<media>> deserialize_packets_view(
    const uint8_t* data,
    size_t size);

} // namespace spdl::core
