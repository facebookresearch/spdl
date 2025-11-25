/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <libspdl/core/frames.h>
#include <libspdl/core/generator.h>
#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <vector>

namespace spdl::core::detail {
////////////////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> get_filters();

////////////////////////////////////////////////////////////////////////////////
// FilterGraphImpl
////////////////////////////////////////////////////////////////////////////////

class FilterGraphImpl {
  AVFilterGraphPtr filter_graph_;
  std::map<std::string, AVFilterContext*> inputs_;
  std::map<std::string, AVFilterContext*> outputs_;

 private:
  template <MediaType media_type>
  FramesPtr<media_type> get_frames(AVFilterContext* filter_ctx);

 public:
  explicit FilterGraphImpl(const std::string& filter_desc);

  void add_frames(const std::string& name, const std::vector<AVFrame*>& frames);
  void add_frames(const std::vector<AVFrame*>& frames);

  void flush();

  std::string dump() const;

  AnyFrames get_frames(const std::string& name);
  AnyFrames get_frames();

  // Public, but internal use only for now
  Generator<AVFramePtr> filter(AVFrame*);
  Rational get_src_time_base() const;
  Rational get_sink_time_base() const;
};

} // namespace spdl::core::detail
