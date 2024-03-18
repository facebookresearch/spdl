#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

struct AVCodecParameters;
struct AVPacket;

namespace spdl::core {

template <MediaType media_type>
struct DemuxedPackets;

using AudioPackets = DemuxedPackets<MediaType::Audio>;
using VideoPackets = DemuxedPackets<MediaType::Video>;
using ImagePackets = DemuxedPackets<MediaType::Image>;

template <MediaType media_type>
using PacketsPtr = std::unique_ptr<DemuxedPackets<media_type>>;

using AudioPacketsPtr = PacketsPtr<MediaType::Audio>;
using VideoPacketsPtr = PacketsPtr<MediaType::Video>;
using ImagePacketsPtr = PacketsPtr<MediaType::Image>;

// Packets are generated in C++, passed to Python and back to C++ again.
// But we want to be able to release the memory in C++.
// To do this, we need to take the ownership of DemuxedPackets instances.
// So we use wrapper to disconnect the Python object and the lifetime of
// the actual DemuxedPackets.

template <MediaType media_type>
class PacketsWrapper;

template <MediaType media_type>
using PacketsWrapperPtr = std::shared_ptr<PacketsWrapper<media_type>>;

using AudioPacketsWrapperPtr = PacketsWrapperPtr<MediaType::Audio>;
using VideoPacketsWrapperPtr = PacketsWrapperPtr<MediaType::Video>;
using ImagePacketsWrapperPtr = PacketsWrapperPtr<MediaType::Image>;

// Struct passed from IO thread pool to decoder thread pool.
// Similar to FFmpegFrames, AVFrame pointers are bulk released.
// It contains suffiient information to build decoder via AVStream*.
template <MediaType media_type>
class DemuxedPackets {
 public:
  uint64_t id;
  // Source information
  std::string src;
  std::tuple<double, double> timestamp;

  //
  AVCodecParameters* codecpar = nullptr;
  Rational time_base = {0, 1};

  // frame rate for video
  Rational frame_rate = {0, 1};

 private:
  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

 public:
  DemuxedPackets(
      std::string src,
      std::tuple<double, double> timestamp,
      AVCodecParameters* codecpar,
      Rational time_base,
      Rational frame_rate);

  // Destructor releases AVPacket* resources
  ~DemuxedPackets();
  // No copy/move constructors
  DemuxedPackets(const DemuxedPackets&) = delete;
  DemuxedPackets& operator=(const DemuxedPackets&) = delete;
  DemuxedPackets(DemuxedPackets&& other) noexcept = delete;
  DemuxedPackets& operator=(DemuxedPackets&& other) noexcept = delete;

  void push(AVPacket*);
  size_t num_packets() const;
  const std::vector<AVPacket*>& get_packets() const;
  const char* get_media_format_name() const;
};

template <MediaType media_type>
class PacketsWrapper {
  PacketsPtr<media_type> packets;

 public:
  PacketsWrapper(PacketsPtr<media_type>&& p) : packets(std::move(p)){};

  PacketsPtr<media_type> unwrap() {
    return std::move(packets);
  }

  const PacketsPtr<media_type>& get_packets() const {
    return packets;
  }
};

template <MediaType media_type>
PacketsWrapperPtr<media_type> wrap(PacketsPtr<media_type>&& packets) {
  return std::make_shared<PacketsWrapper<media_type>>(std::move(packets));
}

} // namespace spdl::core
