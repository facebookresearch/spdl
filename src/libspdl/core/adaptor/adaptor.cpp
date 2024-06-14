#include <libspdl/core/adaptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include <memory>

using namespace spdl::core::detail;

namespace spdl::core {
namespace {
class BasicInterface : public DataInterface {
  detail::AVFormatInputContextPtr fmt_ctx;

 public:
  BasicInterface(std::string_view url, const DemuxConfig& dmx_cfg)
      : fmt_ctx(get_input_format_ctx(
            std::string{url},
            dmx_cfg.format,
            dmx_cfg.format_options)) {}
  ~BasicInterface() override = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};
} // namespace
std::unique_ptr<DataInterface> SourceAdaptor::get(
    std::string_view url,
    const DemuxConfig& dmx_cfg) const {
  return std::unique_ptr<DataInterface>(new BasicInterface(url, dmx_cfg));
}
} // namespace spdl::core
