#pragma once

#include <libspdl/core/adoptor/base.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <folly/experimental/coro/AsyncGenerator.h>
#include <folly/experimental/coro/Task.h>

#include <string>
#include <tuple>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

namespace spdl::core::detail {

struct PackagedAVPackets;

folly::coro::AsyncGenerator<std::unique_ptr<PackagedAVPackets>> stream_demux(
    const enum MediaType type,
    const std::string src,
    const std::vector<std::tuple<double, double>> timestamps,
    std::shared_ptr<SourceAdoptor> adoptor,
    const IOConfig io_cfg);

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_packets(
    std::unique_ptr<PackagedAVPackets> packets,
    const DecodeConfig cfg = {},
    const std::string filter_desc = {});

//////////////////////////////////////////////////////////////////////////////
// PackagedAVPackets
//////////////////////////////////////////////////////////////////////////////
// Struct passed from IO thread pool to decoder thread pool.
// Similar to FrameContainer, AVFrame pointers are bulk released.
// It contains suffiient information to build decoder via AVStream*.
struct PackagedAVPackets {
  uint64_t id;
  // Source information
  std::string src;
  std::tuple<double, double> timestamp;

  //
  AVCodecParameters* codecpar = nullptr;
  AVRational time_base = {0, 1};

  // frame rate for video
  AVRational frame_rate = {0, 1};

  // Sliced raw packets
  std::vector<AVPacket*> packets = {};

  PackagedAVPackets(
      std::string src,
      std::tuple<double, double> timestamp,
      AVCodecParameters* codecpar,
      AVRational time_base,
      AVRational frame_rate);

  // No copy constructors
  PackagedAVPackets(const PackagedAVPackets&) = delete;
  PackagedAVPackets& operator=(const PackagedAVPackets&) = delete;
  // Move constructor to support AsyncGenerator
  PackagedAVPackets(PackagedAVPackets&& other) noexcept;
  PackagedAVPackets& operator=(PackagedAVPackets&& other) noexcept;
  // Destructor releases AVPacket* resources
  ~PackagedAVPackets();
};

} // namespace spdl::core::detail
