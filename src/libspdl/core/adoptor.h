#pragma once

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

struct AVFormatContext;

namespace spdl::core {

struct SourceAdoptor;

using SourceAdoptorPtr = std::shared_ptr<SourceAdoptor>;

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
  virtual DataInterface* get(std::string_view url, const IOConfig& io_cfg)
      const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Basic
////////////////////////////////////////////////////////////////////////////////

// Basic adoptor. Optionally modifies the source with prefix.
// The resulting source indicator is passed to FFmpeg directly.
struct BasicAdoptor : public SourceAdoptor {
  const std::optional<std::string> prefix;

  BasicAdoptor(const std::optional<std::string>& prefix = std::nullopt);

  // note; buffer_size is not used.
  DataInterface* get(std::string_view url, const IOConfig& io_cfg)
      const override;
};

////////////////////////////////////////////////////////////////////////////////
// MMap
////////////////////////////////////////////////////////////////////////////////

struct MMapAdoptor : public SourceAdoptor {
  const std::optional<std::string> prefix;

  MMapAdoptor(const std::optional<std::string>& prefix = std::nullopt);
  DataInterface* get(std::string_view url, const IOConfig& io_cfg)
      const override;
};

////////////////////////////////////////////////////////////////////////////////
// Bytes
////////////////////////////////////////////////////////////////////////////////
struct BytesAdoptor : public SourceAdoptor {
  DataInterface* get(std::string_view data, const IOConfig& io_cfg)
      const override;
};

} // namespace spdl::core
