#pragma once

#include <libspdl/core/frames.h>
#include <libspdl/core/interface/base.h>
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

folly::coro::AsyncGenerator<PackagedAVPackets&&> stream_demux(
    const enum MediaType type,
    const std::string src,
    const std::shared_ptr<SourceAdoptor> adoptor,
    const std::vector<std::tuple<double, double>> timestamps);

folly::coro::Task<std::unique_ptr<FrameContainer>> decode_packets(
    PackagedAVPackets&& packets,
    const std::string filter_desc,
    const DecodeConfig cfg = {});

//////////////////////////////////////////////////////////////////////////////
// PackagedAVPackets
//////////////////////////////////////////////////////////////////////////////
// Struct passed from IO thread pool to decoder thread pool.
// Similar to FrameContainer, AVFrame pointers are bulk released.
// It contains suffiient information to build decoder via AVStream*.
struct PackagedAVPackets {
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
      std::string src_,
      std::tuple<double, double> timestamp_,
      AVCodecParameters* codecpar_,
      AVRational time_base_,
      AVRational frame_rate_);

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
