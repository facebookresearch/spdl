#include <libspdl/detail/ffmpeg/ctx_utils.h>
#include <libspdl/detail/ffmpeg/wrappers.h>
#include <libspdl/detail/interface/mmap.h>
#include <libspdl/interface.h>
#include <memory>

using namespace spdl::detail;

namespace spdl {
namespace {

class Native : public DataProvider {
  AVFormatInputContextPtr fmt_ctx;

 public:
  Native(AVFormatInputContextPtr&& p) : fmt_ctx(std::move(p)) {}
  ~Native() = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};

class MemoryMap : public DataProvider {
  MemoryMappedFile file;
  AVIOContextPtr io_ctx;
  AVFormatInputContextPtr fmt_ctx;

 public:
  MemoryMap(
      const std::string url,
      const std::optional<std::string>& format,
      const std::optional<OptionDict>& format_options,
      int buffer_size)
      : file(std::move(url)),
        io_ctx(get_io_ctx(
            &file,
            buffer_size,
            MemoryMappedFile::read_packet,
            MemoryMappedFile::seek)),
        fmt_ctx(get_input_format_ctx(io_ctx.get(), format, format_options)) {}
  ~MemoryMap() = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};

} // namespace

std::unique_ptr<DataProvider> get_data_provider(
    const std::string& url,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options,
    int buffer_size) {
  if (url.starts_with("mmap://")) {
    return std::unique_ptr<DataProvider>{
        new MemoryMap{url.substr(7), format, format_options, buffer_size}};
  }
  auto fmt_ctx = get_input_format_ctx(url, format, format_options);
  return std::unique_ptr<DataProvider>{new Native{std::move(fmt_ctx)}};
}

} // namespace spdl
