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

template <MediaType media_type>
using PacketsPtr = std::unique_ptr<DemuxedPackets<media_type>>;

// Struct passed from IO thread pool to decoder thread pool.
// Similar to FFmpegFrames, AVFrame pointers are bulk released.
// It contains suffiient information to build decoder via AVStream*.
template <MediaType media_type>
struct DemuxedPackets {
  uint64_t id;
  // Source information
  std::string src;
  std::tuple<double, double> timestamp;

  //
  AVCodecParameters* codecpar = nullptr;
  Rational time_base = {0, 1};

  // frame rate for video
  Rational frame_rate = {0, 1};

  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

  DemuxedPackets(
      std::string src,
      std::tuple<double, double> timestamp,
      AVCodecParameters* codecpar,
      Rational time_base,
      Rational frame_rate);

  // No copy constructors
  DemuxedPackets(const DemuxedPackets&) = delete;
  DemuxedPackets& operator=(const DemuxedPackets&) = delete;
  // Move constructor to support AsyncGenerator
  DemuxedPackets(DemuxedPackets&& other) noexcept;
  DemuxedPackets& operator=(DemuxedPackets&& other) noexcept;
  // Destructor releases AVPacket* resources
  ~DemuxedPackets();
};
} // namespace spdl::core
