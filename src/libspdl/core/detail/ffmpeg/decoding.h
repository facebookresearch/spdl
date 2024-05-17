#pragma once

#include <libspdl/core/decoding.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <deque>

namespace spdl::core {
namespace detail {

struct Decoder;

// Helper structure that converts decoding operation (while loop) into
// iterator.
struct IterativeDecoding {
  struct Sentinel {};

  static constexpr Sentinel sentinel{};

  Decoder* decoder;
  AVPacket* packet;
  bool flush_null;

  struct Ite {
    Decoder* decoder;
    bool completed = false;
    bool null_flushed;
    AVFramePtr next_ret{};

    bool operator!=(const Sentinel&);

    Ite(Decoder*, AVPacket*, bool flush_null);

    Ite& operator++();

    AVFramePtr operator*();

   private:
    void fill_next();
  };

  IterativeDecoding(Decoder*, AVPacket*, bool flush_null);

  Ite begin();
  const Sentinel& end();
};

// Wraps AVCodecContextPtr and provide convenient methods
struct Decoder {
  AVCodecContextPtr codec_ctx;

  Decoder(
      AVCodecParameters*,
      Rational time_base,
      const std::optional<DecodeConfig>& cfg = std::nullopt);

  IterativeDecoding decode(AVPacket*, bool flush_null = false);

 private:
  void add_packet(AVPacket*);
  int get_frame(AVFrame* output);

  friend class IterativeDecoding;
};

} // namespace detail

using spdl::core::detail::FilterGraph;

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder<media_type>::Impl {
  PacketsPtr<media_type> packets;
  detail::Decoder decoder;
  std::optional<FilterGraph> filter_graph;

  std::deque<FFmpegFramesPtr<media_type>> carry_overs;
  int packet_index = 0;
  Impl(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig> cfg,
      const std::string filter_desc);

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

} // namespace spdl::core
