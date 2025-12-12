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

/// Shared pointer to a SourceAdaptor.
using SourceAdaptorPtr = std::shared_ptr<SourceAdaptor>;

////////////////////////////////////////////////////////////////////////////////
// DataInterface
////////////////////////////////////////////////////////////////////////////////

/// Interface for accessing media data sources.
///
/// DataInterface encapsulates FFmpeg's AVFormatContext and AVIOContext,
/// managing their lifetimes and providing a uniform interface for different
/// data sources (files, URLs, memory buffers).
///
/// The demuxer implementation uses AVFormatContext::url for error reporting,
/// so implementations should ensure this field is populated.
struct DataInterface {
  /// Virtual destructor.
  virtual ~DataInterface() = default;

  /// Get the FFmpeg format context.
  ///
  /// @return Pointer to AVFormatContext.
  virtual AVFormatContext* get_fmt_ctx() = 0;
};

/// Unique pointer to a DataInterface.
using DataInterfacePtr = std::unique_ptr<DataInterface>;

////////////////////////////////////////////////////////////////////////////////
// Adaptor
////////////////////////////////////////////////////////////////////////////////

/// Adaptor for custom data sources.
///
/// SourceAdaptor optionally modifies the input resource indicator and
/// creates a DataInterface to access the actual data source.
struct SourceAdaptor {
  /// Virtual destructor.
  virtual ~SourceAdaptor() = default;

  /// Create a data interface for the given URL.
  ///
  /// @param url Resource URL or identifier.
  /// @param dmx_cfg Demuxer configuration.
  /// @param name Optional custom name for the source (used in error messages).
  /// @return DataInterface for accessing the resource.
  virtual DataInterfacePtr get_interface(
      std::string_view url,
      const DemuxConfig& dmx_cfg,
      const std::optional<std::string>& name = std::nullopt) const;
};

////////////////////////////////////////////////////////////////////////////////
// Bytes
////////////////////////////////////////////////////////////////////////////////

/// Adaptor for in-memory data.
///
/// BytesAdaptor creates a data interface for complete, contiguous media data
/// stored in memory (not partial/streaming data).
struct BytesAdaptor : public SourceAdaptor {
  /// Create a data interface for in-memory data.
  ///
  /// @param data String view of the media data.
  /// @param dmx_cfg Demuxer configuration.
  /// @return DataInterface for accessing the memory buffer.
  DataInterfacePtr get_interface(
      std::string_view data,
      const DemuxConfig& dmx_cfg,
      const std::optional<std::string>& name = std::nullopt) const override;
};

} // namespace spdl::core
