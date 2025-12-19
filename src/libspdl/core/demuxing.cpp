/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>

#include "libspdl/common/tracing.h"
#include "libspdl/core/detail/ffmpeg/demuxer.h"
#include "libspdl/core/detail/ffmpeg/logging.h"

namespace spdl::core {
namespace detail {
namespace {
DataInterfacePtr get_interface(
    const std::string_view src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& name) {
  if (!adaptor) {
    thread_local auto p = std::make_shared<SourceAdaptor>();
    return p->get_interface(src, dmx_cfg.value_or(DemuxConfig{}), name);
  }
  return adaptor->get_interface(src, dmx_cfg.value_or(DemuxConfig{}), name);
}

DataInterfacePtr get_in_memory_interface(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& name) {
  thread_local SourceAdaptorPtr adaptor = std::make_shared<BytesAdaptor>();
  return adaptor->get_interface(data, dmx_cfg.value_or(DemuxConfig{}), name);
}
} // namespace
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// StreamingDemuxer
////////////////////////////////////////////////////////////////////////////////
StreamingDemuxer::StreamingDemuxer(
    detail::DemuxerImpl* p,
    const std::set<int>& stream_indices,
    int num_packets,
    double duration)
    : gen_(p->streaming_demux(stream_indices, num_packets, duration)) {}

bool StreamingDemuxer::done() {
  return !bool(gen_);
}

std::map<int, AnyPackets> StreamingDemuxer::next() {
  return gen_();
}

////////////////////////////////////////////////////////////////////////////////
// Demuxer
////////////////////////////////////////////////////////////////////////////////

Demuxer::Demuxer(DataInterfacePtr di)
    : pImpl_(new detail::DemuxerImpl(std::move(di))) {}

Demuxer::~Demuxer() {
  if (pImpl_) {
    delete pImpl_;
  }
}

bool Demuxer::has_audio() const {
  return pImpl_->has_audio();
}

template <MediaType media>
Codec<media> Demuxer::get_default_codec() const {
  return pImpl_->get_default_codec<media>();
}
template Codec<MediaType::Audio> Demuxer::get_default_codec<MediaType::Audio>()
    const;
template Codec<MediaType::Video> Demuxer::get_default_codec<MediaType::Video>()
    const;
template Codec<MediaType::Image> Demuxer::get_default_codec<MediaType::Image>()
    const;

template <MediaType media>
int Demuxer::get_default_stream_index() const {
  return pImpl_->get_default_stream_index<media>();
}

template int Demuxer::get_default_stream_index<MediaType::Audio>() const;
template int Demuxer::get_default_stream_index<MediaType::Video>() const;
template int Demuxer::get_default_stream_index<MediaType::Image>() const;

template <MediaType media>
PacketsPtr<media> Demuxer::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf) {
  return pImpl_->demux_window<media>(window, bsf);
}

template PacketsPtr<MediaType::Audio> Demuxer::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Video> Demuxer::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf);

template PacketsPtr<MediaType::Image> Demuxer::demux_window(
    const std::optional<TimeWindow>& window,
    const std::optional<std::string>& bsf);

StreamingDemuxerPtr Demuxer::streaming_demux(
    const std::set<int>& indices,
    int num_packets,
    double duration) {
  return std::make_unique<StreamingDemuxer>(
      pImpl_, indices, num_packets, duration);
}

DemuxerPtr make_demuxer(
    const std::string& src,
    const SourceAdaptorPtr& adaptor,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& name) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer>(
      detail::get_interface(src, adaptor, dmx_cfg, name));
}

DemuxerPtr make_demuxer(
    const std::string_view data,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& name) {
  TRACE_EVENT("demuxing", "make_demuxer");
  return std::make_unique<Demuxer>(
      detail::get_in_memory_interface(data, dmx_cfg, name));
}

} // namespace spdl::core
