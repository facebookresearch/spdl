#include <libspdl/core/interface/basic.h>

#include <libspdl/core/detail/ffmpeg/ctx_utils.h>

#include <folly/logging/xlog.h>

#include <memory>

using namespace spdl::core::detail;

namespace spdl::core {

class BasicInterface : public DataInterface {
  detail::AVFormatInputContextPtr fmt_ctx;

 public:
  BasicInterface(
      const std::string& url,
      const std::optional<std::string>& format,
      const std::optional<OptionDict>& format_options)
      : fmt_ctx(get_input_format_ctx_ptr(url, format, format_options)) {}
  ~BasicInterface() = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};

BasicAdoptor::BasicAdoptor(
    const std::optional<std::string>& prefix_,
    const std::optional<std::string>& format_,
    const std::optional<OptionDict>& format_options_)
    : prefix(prefix_), format(format_), format_options(format_options_) {}

void* BasicAdoptor::get(const std::string& url) {
  return new BasicInterface(
      prefix ? prefix.value() + url : url, format, format_options);
}
} // namespace spdl::core
