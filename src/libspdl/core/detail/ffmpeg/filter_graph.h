/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <libspdl/core/generator.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <vector>

namespace spdl::core::detail {
////////////////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> get_filters();

// Wrap AVFilterGraphPtr to provide convenient methods
class FilterGraph {
  AVFilterGraphPtr graph;

 public:
  explicit FilterGraph(AVFilterGraphPtr&& g) : graph(std::move(g)) {}

  Generator<AVFramePtr> filter(AVFrame*);

  Rational get_src_time_base() const;
  Rational get_sink_time_base() const;
};

FilterGraph get_audio_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

FilterGraph get_video_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx,
    Rational frame_rate);

FilterGraph get_image_filter(
    const std::string& filter_description,
    AVCodecContext* codec_ctx);

FilterGraph get_image_enc_filter(
    int src_width,
    int src_height,
    AVPixelFormat src_fmt,
    int enc_width,
    int enc_height,
    const std::optional<std::string>& scale_algo,
    AVPixelFormat enc_fmt,
    const std::optional<std::string>& filter_desc);

template <MediaType media_type>
FilterGraph get_filter(
    AVCodecContext* codec_ctx,
    const std::string& filter_desc,
    std::optional<Rational> frame_rate) {
  if constexpr (media_type == MediaType::Audio) {
    return get_audio_filter(filter_desc, codec_ctx);
  }
  if constexpr (media_type == MediaType::Video) {
    return get_video_filter(filter_desc, codec_ctx, *frame_rate);
  }
  if constexpr (media_type == MediaType::Image) {
    return get_image_filter(filter_desc, codec_ctx);
  }
}

template <MediaType media_type>
std::optional<FilterGraph> get_filter(
    AVCodecContext* codec_ctx,
    const std::optional<std::string>& filter_desc,
    std::optional<Rational> frame_rate) {
  if (filter_desc) {
    return get_filter<media_type>(codec_ctx, *filter_desc, frame_rate);
  } else {
    return std::nullopt;
  }
}

} // namespace spdl::core::detail
