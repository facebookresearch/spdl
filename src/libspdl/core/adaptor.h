/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <string>
#include <string_view>

struct AVFormatContext;

namespace spdl::core {

struct SourceAdaptor;

using SourceAdaptorPtr = std::shared_ptr<SourceAdaptor>;

////////////////////////////////////////////////////////////////////////////////
// DataInterface
////////////////////////////////////////////////////////////////////////////////

// DataInterface serves two purposes:
// 1. It encapsulates the AVFormatContext, AVIOContext and other
//    objects so that their lifetimes are aligned.
//    These are consumed in demuxer functions.
// 2. Abstract away the difference between the interfaces. Each
//    interface can require additional/specific implementation to
//    talk to the actual data source. Such things are distilled to
//    merely AVFormatContext*.
struct DataInterface {
  virtual ~DataInterface() = default;
  virtual AVFormatContext* get_fmt_ctx() = 0;
  virtual std::string get_src() const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Adaptor
////////////////////////////////////////////////////////////////////////////////
// Adaptor optionally modifies the intput resource indicator, and create
// DataInterface from the result.
struct SourceAdaptor {
  virtual ~SourceAdaptor() = default;

  virtual std::unique_ptr<DataInterface> get_interface(
      std::string_view url,
      const DemuxConfig& dmx_cfg) const;
};

////////////////////////////////////////////////////////////////////////////////
// Bytes
////////////////////////////////////////////////////////////////////////////////
struct BytesAdaptor : public SourceAdaptor {
  std::unique_ptr<DataInterface> get_interface(
      std::string_view data,
      const DemuxConfig& dmx_cfg) const override;
};

} // namespace spdl::core
