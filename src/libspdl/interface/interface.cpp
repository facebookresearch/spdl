#include <libspdl/ffmpeg/ctx_utils.h>
#include <libspdl/ffmpeg/wrappers.h>
#include <libspdl/interface/interface.h>
#include <libspdl/interface/mmap.h>
#include <memory>

namespace spdl::interface {
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
      const std::string_view url,
      const std::optional<OptionDict> options,
      const std::optional<std::string> format,
      int buffer_size)
      : file(url),
        io_ctx(get_io_ctx(
            &file,
            buffer_size,
            MemoryMappedFile::read_packet,
            MemoryMappedFile::seek)),
        fmt_ctx(get_input_format_ctx(io_ctx.get(), options, format)) {}
  ~MemoryMap() = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};

} // namespace

std::unique_ptr<DataProvider> get_data_provider(
    const std::string_view url,
    const std::optional<OptionDict> options,
    const std::optional<std::string> format,
    int buffer_size) {
  if (url.starts_with("mmap://")) {
    return std::unique_ptr<DataProvider>{
        new MemoryMap{url.substr(7), options, format, buffer_size}};
  }
  auto fmt_ctx = get_input_format_ctx(url, options, format);
  return std::unique_ptr<DataProvider>{new Native{std::move(fmt_ctx)}};
}

} // namespace spdl::interface
