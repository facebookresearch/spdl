#pragma once

#include <libspdl/core/interface/base.h>

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>

namespace spdl::core {

// Basic adoptor. Optionally modifies the source with prefix.
// The resulting source indicator is passed to FFmpeg directly.
struct BasicAdoptor : public SourceAdoptor {
  std::optional<std::string> prefix;

  BasicAdoptor(const std::optional<std::string>& prefix = std::nullopt);

  // note; buffer_size is not used.
  void* get(const std::string& url, const IOConfig& io_cfg) override;
};

} // namespace spdl::core
