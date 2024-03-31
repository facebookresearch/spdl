#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
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
};

////////////////////////////////////////////////////////////////////////////////
// Adaptor
////////////////////////////////////////////////////////////////////////////////
// Adaptor optionally modifies the intput resource indicator, and create
// DataInterface from the result.
struct SourceAdaptor {
  virtual ~SourceAdaptor() = default;

  // This returns a pointer to DataInterface classes, but for the sake of
  // exposing this via PyBind11, we use void*
  virtual std::unique_ptr<DataInterface> get(
      std::string_view url,
      const IOConfig& io_cfg) const;
};

////////////////////////////////////////////////////////////////////////////////
// MMap
////////////////////////////////////////////////////////////////////////////////
struct MMapAdaptor : public SourceAdaptor {
  std::unique_ptr<DataInterface> get(
      std::string_view url,
      const IOConfig& io_cfg) const override;
};

////////////////////////////////////////////////////////////////////////////////
// Bytes
////////////////////////////////////////////////////////////////////////////////
struct BytesAdaptor : public SourceAdaptor {
  std::unique_ptr<DataInterface> get(
      std::string_view data,
      const IOConfig& io_cfg) const override;
};

} // namespace spdl::core
