#pragma once

#include <memory>

struct AVIOContext;
struct AVFormatContext;

namespace spdl::core {

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
// Adoptor
////////////////////////////////////////////////////////////////////////////////

// Adoptor optionally modifies the intput resource indicator, and create
// DataInterface from the result.
struct SourceAdoptor {
  virtual ~SourceAdoptor() = default;
  virtual std::unique_ptr<DataInterface> get(const std::string& url) = 0;
};

} // namespace spdl::core
