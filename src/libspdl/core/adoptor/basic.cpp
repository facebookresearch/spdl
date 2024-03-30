#include <libspdl/core/adoptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include <folly/logging/xlog.h>

#include <memory>

using namespace spdl::core::detail;

namespace spdl::core {

class BasicInterface : public DataInterface {
  detail::AVFormatInputContextPtr fmt_ctx;

 public:
  BasicInterface(std::string url, const IOConfig& io_cfg)
      : fmt_ctx(get_input_format_ctx_ptr(
            std::move(url),
            io_cfg.format,
            io_cfg.format_options)) {}
  ~BasicInterface() = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};

BasicAdoptor::BasicAdoptor(const std::optional<std::string>& prefix_)
    : prefix(prefix_) {}

DataInterface* BasicAdoptor::get(std::string_view url, const IOConfig& io_cfg)
    const {
  return new BasicInterface(
      prefix ? prefix.value() + std::string(url) : std::string(url), io_cfg);
}
} // namespace spdl::core
