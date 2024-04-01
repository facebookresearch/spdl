#include <libspdl/core/adaptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include <folly/logging/xlog.h>

#include <memory>

using namespace spdl::core::detail;

namespace spdl::core {

class BasicInterface : public DataInterface {
  detail::AVFormatInputContextPtr fmt_ctx;

 public:
  BasicInterface(std::string_view url, const IOConfig& io_cfg)
      : fmt_ctx(get_input_format_ctx(
            std::string{url},
            io_cfg.format,
            io_cfg.format_options)) {}
  ~BasicInterface() = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};

std::unique_ptr<DataInterface> SourceAdaptor::get(
    std::string_view url,
    const IOConfig& io_cfg) const {
  return std::unique_ptr<DataInterface>(new BasicInterface(url, io_cfg));
}
} // namespace spdl::core
