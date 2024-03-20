#pragma once

#include <libspdl/core/adoptor/custom.h>

#include <string>

namespace spdl::core {
struct BytesAdoptor : public SourceAdoptor {
 public:
  BytesAdoptor() = default;
  ~BytesAdoptor() = default;

  void* get(const std::string& data, const IOConfig& io_cfg) const override;
};
} // namespace spdl::core
