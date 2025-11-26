/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/filter_graph.h>

#include "libspdl/core/detail/ffmpeg/filter_graph.h"

namespace spdl::core {
FilterGraph::FilterGraph(const std::string& filter_desc)
    : pImpl_(new detail::FilterGraphImpl(filter_desc)) {}

void FilterGraph::add_frames(
    const AnyFrames& frames,
    const std::optional<std::string>& name) {
  const auto f = std::visit([&](auto& v) { return v->get_frames(); }, frames);
  if (name) {
    pImpl_->add_frames(*name, f);
  } else {
    pImpl_->add_frames(f);
  }
}

void FilterGraph::flush() {
  return pImpl_->flush();
}

std::optional<AnyFrames> FilterGraph::get_frames(
    const std::optional<std::string>& name) {
  AnyFrames frames = name ? pImpl_->get_frames(*name) : pImpl_->get_frames();
  // Check if the frames object is empty by visiting the variant and checking
  // the number of frames
  bool is_empty = std::visit(
      [](const auto& f) { return f->get_num_frames() == 0; }, frames);
  if (is_empty) {
    return std::nullopt;
  }
  return frames;
}

std::string FilterGraph::dump() const {
  return pImpl_->dump();
}

FilterGraph::~FilterGraph() {
  delete pImpl_;
}

FilterGraphPtr make_filter_graph(const std::string& filter_desc) {
  return std::make_unique<FilterGraph>(filter_desc);
}

} // namespace spdl::core
