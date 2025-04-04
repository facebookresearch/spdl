/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/frames.h>

#include <memory>
#include <variant>

namespace spdl::core {
namespace detail {
class FilterGraphImpl;
}

class FilterGraph {
  detail::FilterGraphImpl* pImpl;

 public:
  FilterGraph(
      const std::string& filter_desc,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs);
  ~FilterGraph();
  FilterGraph(const FilterGraph&) = delete;
  FilterGraph& operator=(const FilterGraph&) = delete;
  FilterGraph(FilterGraph&&) = delete;
  FilterGraph& operator=(FilterGraph&&) = delete;

  template <MediaType media>
  void add_frames(const std::string& name, FramesPtr<media> frames);

  // void add_frames(const std::map<std::string, AnyFrames> frames);
  void flush();
  AnyFrames get_frames(const std::string& name);

  std::string dump() const;
};

using FilterGraphPtr = std::unique_ptr<FilterGraph>;

FilterGraphPtr make_filter_graph(
    const std::string& filter_desc,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs);

} // namespace spdl::core
