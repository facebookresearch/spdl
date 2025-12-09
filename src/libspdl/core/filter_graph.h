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

/// FFmpeg filter graph for processing frames.
///
/// FilterGraph applies complex filtering operations to audio or video frames
/// using FFmpeg's libavfilter. Filters can be chained and configured using
/// FFmpeg filter syntax.
class FilterGraph {
  detail::FilterGraphImpl* pImpl_;

 public:
  /// Construct a filter graph from a filter description.
  ///
  /// @param filter_desc FFmpeg filter description string.
  explicit FilterGraph(const std::string& filter_desc);

  /// Destructor.
  ~FilterGraph();

  /// Deleted copy constructor.
  FilterGraph(const FilterGraph&) = delete;

  /// Deleted copy assignment operator.
  FilterGraph& operator=(const FilterGraph&) = delete;

  /// Deleted move constructor.
  FilterGraph(FilterGraph&&) = delete;

  /// Deleted move assignment operator.
  FilterGraph& operator=(FilterGraph&&) = delete;

  /// Add frames to the filter graph for processing.
  ///
  /// @param frames Frames to add (audio, video, or image).
  /// @param name Optional input pad name.
  void add_frames(
      const AnyFrames& frames,
      const std::optional<std::string>& name);

  /// Flush the filter graph to process remaining frames.
  void flush();

  /// Get filtered frames from the filter graph.
  ///
  /// @param name Optional output pad name.
  /// @return Filtered frames, or std::nullopt if no frames are available.
  std::optional<AnyFrames> get_frames(const std::optional<std::string>& name);

  /// Dump the filter graph configuration for debugging.
  ///
  /// @return String representation of the filter graph.
  std::string dump() const;
};

/// Unique pointer to a FilterGraph instance.
using FilterGraphPtr = std::unique_ptr<FilterGraph>;

/// Create a filter graph from a filter description.
///
/// @param filter_desc FFmpeg filter description string.
/// @return FilterGraph instance.
FilterGraphPtr make_filter_graph(const std::string& filter_desc);

} // namespace spdl::core
