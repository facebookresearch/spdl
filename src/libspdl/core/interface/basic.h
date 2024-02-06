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
  std::optional<std::string> format;
  std::optional<OptionDict> format_options;

  BasicAdoptor(
      const std::optional<std::string>& prefix = std::nullopt,
      const std::optional<std::string>& format = std::nullopt,
      const std::optional<OptionDict>& format_options = std::nullopt);
  void* get(const std::string& url) override;
};

} // namespace spdl::core
