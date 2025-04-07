/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/adaptor.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/packets.h>

#include "libspdl/core/detail/ffmpeg/bsf.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <set>

namespace spdl::core::detail {

class DemuxerImpl {
  DataInterfacePtr di;
  AVFormatContext* fmt_ctx = nullptr;

  Generator<AVPacketPtr>
  demux_window(AVStream* stream, const double end, std::optional<BSFImpl>& bsf);

 public:
  explicit DemuxerImpl(DataInterfacePtr di);
  DemuxerImpl(const DemuxerImpl&) = delete;
  DemuxerImpl& operator=(const DemuxerImpl&) = delete;
  DemuxerImpl(DemuxerImpl&&) = delete;
  DemuxerImpl& operator=(DemuxerImpl&&) = delete;
  ~DemuxerImpl();

  template <MediaType media>
  Codec<media> get_default_codec() const;

  template <MediaType media>
  int get_default_stream_index() const;

  bool has_audio() const;

 private:
  Generator<AVPacketPtr> demux();

 public:
  template <MediaType media>
  PacketsPtr<media> demux_window(
      const std::optional<std::tuple<double, double>>& window,
      const std::optional<std::string>& bsf);

  int get_default_stream_index(MediaType media) const;

  Generator<std::map<int, AnyPackets>> streaming_demux(
      const std::set<int> stream_indices,
      int num_packets);
};

} // namespace spdl::core::detail
