/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/ffmpeg/demuxer.h"
#include "libspdl/core/detail/ffmpeg/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

namespace spdl::core {
namespace detail {
DataInterfacePtr get_interface(
    const std::string_view src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg) {
  if (!adaptor) {
    thread_local auto p = std::make_shared<SourceAdaptor>();
    return p->get_interface(src, dmx_cfg.value_or(DemuxConfig{}));
  }
  return adaptor->get_interface(src, dmx_cfg.value_or(DemuxConfig{}));
}

DataInterfacePtr get_in_memory_interface(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  thread_local SourceAdaptorPtr adaptor = std::make_shared<BytesAdaptor>();
  return get_interface(data, adaptor, dmx_cfg);
}
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// StreamingDemuxer
////////////////////////////////////////////////////////////////////////////////
StreamingDemuxer::StreamingDemuxer(
    detail::DemuxerImpl* p,
    int stream_index,
    int num_packets,
    const std::optional<std::string>& bsf)
    : gen(p->streaming_demux(stream_index, num_packets, bsf)) {}

bool StreamingDemuxer::done() {
  return !bool(gen);
}

AnyPackets StreamingDemuxer::next() {
  return gen();
}

////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////

Demuxer::Demuxer(DataInterfacePtr di)
    : pImpl(new detail::DemuxerImpl(std::move(di))){};

Demuxer::~Demuxer() {
  if (pImpl) {
    delete pImpl;
  }
}

bool Demuxer::has_audio() const {
  return pImpl->has_audio();
}

template <MediaType media>
Codec<media> Demuxer::get_default_codec() const {
  return pImpl->get_default_codec<media>();
}
template Codec<MediaType::Audio> Demuxer::get_default_codec<MediaType::Audio>()
    const;
template Codec<MediaType::Video> Demuxer::get_default_codec<MediaType::Video>()
    const;
template Codec<MediaType::Image> Demuxer::get_default_codec<MediaType::Image>()
    const;

template <MediaType media>
PacketsPtr<media> Demuxer::demux_window(
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf) {
  return pImpl->demux_window<media>(window, bsf);
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

StreamingDemuxerPtr Demuxer::stream_demux(
    int num_packets,
    const std::optional<std::string>& bsf) {
  int i = pImpl->get_default_stream_index(MediaType::Video);
  return std::make_unique<StreamingDemuxer>(pImpl, i, num_packets, bsf);
}

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
VideoPacketsPtr apply_bsf(VideoPacketsPtr packets, const std::string& name) {
  if (!packets->codec) {
    throw std::runtime_error("The packets do not have codec.");
  }
  TRACE_EVENT("demuxing", "apply_bsf");
  auto bsf = detail::BitStreamFilter{name, packets->codec->get_parameters()};

  auto ret = std::make_unique<Packets<MediaType::Video>>(
      packets->src,
      VideoCodec{
          bsf.get_output_codec_par(),
          packets->codec->get_time_base(),
          packets->codec->get_frame_rate()},
      packets->timestamp);

  for (auto& packet : packets->pkts.get_packets()) {
    auto filtering = bsf.filter(packet);
    while (filtering) {
      ret->pkts.push(filtering().release());
    }
  }
  return ret;
}

} // namespace spdl::core
