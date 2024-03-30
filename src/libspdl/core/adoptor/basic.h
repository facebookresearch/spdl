#pragma once

#include <libspdl/core/adoptor/base.h>

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>

namespace spdl::core {

// Basic adoptor. Optionally modifies the source with prefix.
// The resulting source indicator is passed to FFmpeg directly.
struct BasicAdoptor : public SourceAdoptor {
  const std::optional<std::string> prefix;

  BasicAdoptor(const std::optional<std::string>& prefix = std::nullopt);

  // note; buffer_size is not used.
  void* get(std::string_view url, const IOConfig& io_cfg) const override;
};

} // namespace spdl::core
