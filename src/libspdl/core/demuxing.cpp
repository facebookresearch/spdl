/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

extern "C" {
#include <libavformat/avformat.h>
}

namespace spdl::core {

// Implemented in core/detail/ffmpeg/demuxing.cpp
namespace detail {
void init_fmt_ctx(AVFormatContext* fmt_ctx);
AVStream* get_stream(AVFormatContext* fmt_ctx, enum MediaType type_);

template <MediaType media_type>
PacketsPtr<media_type> demux_window(
    AVFormatContext* fmt_ctx,
    AVStream* stream,
    const std::optional<std::tuple<double, double>>& window = std::nullopt,
    const std::optional<std::string>& bsf = std::nullopt);

std::unique_ptr<DataInterface> get_interface(
    const std::string_view src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  if (!adaptor) {
    thread_local auto p = std::make_shared<SourceAdaptor>();
    return p->get_interface(src, dmx_cfg.value_or(DemuxConfig{}));
  }
  return adaptor->get_interface(src, dmx_cfg.value_or(DemuxConfig{}));
}

std::unique_ptr<DataInterface> get_in_memory_interface(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  thread_local SourceAdaptorPtr adaptor = std::make_shared<BytesAdaptor>();
  return get_interface(data, adaptor, dmx_cfg);
}
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////

Demuxer::Demuxer(std::unique_ptr<DataInterface> di_)
    : di(std::move(di_)), fmt_ctx(di->get_fmt_ctx()) {
  detail::init_fmt_ctx(fmt_ctx);
};

Demuxer::~Demuxer() {
  TRACE_EVENT("demuxing", "Demuxer::~Demuxer");
  di.reset();
  // Technically, this is not necessary, but doing it here puts
  // the destruction of AVFormatContext under ~StreamingDemuxe, which
  // makes the trace easier to interpret.
}

bool Demuxer::has_audio() {
  for (int i = 0; i < fmt_ctx->nb_streams; ++i) {
    if (AVMEDIA_TYPE_AUDIO == fmt_ctx->streams[i]->codecpar->codec_type) {
      return true;
    }
  }
  return false;
}

template <MediaType media_type>
PacketsPtr<media_type> Demuxer::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf) {
  auto stream = detail::get_stream(fmt_ctx, media_type);
  auto packets = detail::demux_window<media_type>(fmt_ctx, stream, window, bsf);
  if constexpr (media_type == MediaType::Video) {
    auto frame_rate = av_guess_frame_rate(fmt_ctx, stream, nullptr);
    packets->frame_rate = Rational{frame_rate.num, frame_rate.den};
  }
  return packets;
}

template PacketsPtr<MediaType::Audio> Demuxer::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Video> Demuxer::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Image> Demuxer::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf);

DemuxerPtr make_demuxer(
    const std::string& src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer>(
      detail::get_interface(src, adaptor, dmx_cfg));
}

DemuxerPtr make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer>(
      detail::get_in_memory_interface(data, dmx_cfg));
}

////////////////////////////////////////////////////////////////////////////////
// Bit Stream Filtering for NVDEC
////////////////////////////////////////////////////////////////////////////////
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets) {
  // Note
  // FFmpeg's implementation applies BSF to all H264/HEVC formats,
  //
  // https://github.com/FFmpeg/FFmpeg/blob/5e2b0862eb1d408625232b37b7a2420403cd498f/libavcodec/cuviddec.c#L1185-L1191
  //
  // while NVidia SDK samples exclude those with the following substrings in
  // long_name attribute
  //
  //  "QuickTime / MOV", "FLV (Flash Video)", "Matroska / WebM"
  const char* name;
  switch (packets->codecpar->codec_id) {
    case AV_CODEC_ID_H264:
      name = "h264_mp4toannexb";
      break;
    case AV_CODEC_ID_HEVC:
      name = "hevc_mp4toannexb";
      break;
    default:
      return packets;
  }

  TRACE_EVENT("demuxing", "apply_bsf");
  auto bsf = detail::BitStreamFilter{name, packets->codecpar};

  auto ret = std::make_unique<DemuxedPackets<MediaType::Video>>(
      packets->src, bsf.get_output_codec_par(), packets->time_base);
  ret->timestamp = packets->timestamp;
  ret->frame_rate = packets->frame_rate;
  for (auto& packet : packets->get_packets()) {
    auto filtering = bsf.filter(packet);
    while (filtering) {
      ret->push(filtering().release());
    }
  }
  return ret;
}

} // namespace spdl::core
