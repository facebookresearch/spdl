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

  void add_packet(AVPacket*);
  int get_frame(AVFrame* output);
};

} // namespace detail

template <MediaType media_type>
  requires(media_type != MediaType::Image)
struct StreamingDecoder<media_type>::Impl {
  PacketsPtr<media_type> packets;
  detail::Decoder decoder;
  detail::FilterGraph filter_graph;

  int packet_index = 0;

  // Decoding && filtering is usually implemented as nested while loop
  // ```
  // decoder.add_packet(packet);
  // while (decoder_has_frame) {
  //   filter.add_frame(decoder.get_frame());
  //   while (filter_has_frame) {
  //     frame = filter.get_frame();
  //   }
  // }
  // ```
  // To make it interruptable/resumable, we keep two states.
  bool decoder_has_frame = false;
  bool filter_has_frame = false;
  bool completed = false;
  Impl(
      PacketsPtr<media_type> packets,
      const std::optional<DecodeConfig> cfg,
      const std::string filter_desc);

  std::optional<FFmpegFramesPtr<media_type>> decode(int num_frames);
};

} // namespace spdl::core
