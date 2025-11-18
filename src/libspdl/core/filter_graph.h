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
  explicit FilterGraph(const std::string& filter_desc);
  ~FilterGraph();
  FilterGraph(const FilterGraph&) = delete;
  FilterGraph& operator=(const FilterGraph&) = delete;
  FilterGraph(FilterGraph&&) = delete;
  FilterGraph& operator=(FilterGraph&&) = delete;

  void add_frames(
      const AnyFrames& frames,
      const std::optional<std::string>& name);

  void flush();

  std::optional<AnyFrames> get_frames(const std::optional<std::string>& name);

  std::string dump() const;
};

using FilterGraphPtr = std::unique_ptr<FilterGraph>;

FilterGraphPtr make_filter_graph(const std::string& filter_desc);

} // namespace spdl::core
