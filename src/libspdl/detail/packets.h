#pragma once

#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

namespace spdl::detail {

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

} // namespace spdl::detail
