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
FilterGraph::FilterGraph(
    const std::string& filter_desc,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs)
    : pImpl(new detail::FilterGraphImpl(filter_desc, inputs, outputs)) {}

template <MediaType media>
void FilterGraph::add_frames(const std::string& name, FramesPtr<media> frames) {
  pImpl->add_frames(name, frames->get_frames());
}

template void FilterGraph::add_frames(const std::string&, AudioFramesPtr);
template void FilterGraph::add_frames(const std::string&, VideoFramesPtr);
template void FilterGraph::add_frames(const std::string&, ImageFramesPtr);

void FilterGraph::flush() {
  return pImpl->flush();
}

AnyFrames FilterGraph::get_frames(const std::string& name) {
  return pImpl->get_frames(name);
}

std::string FilterGraph::dump() const {
  return pImpl->dump();
}

FilterGraph::~FilterGraph() {
  delete pImpl;
}

FilterGraphPtr make_filter_graph(
    const std::string& filter_desc,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  return std::make_unique<FilterGraph>(filter_desc, inputs, outputs);
}

} // namespace spdl::core
