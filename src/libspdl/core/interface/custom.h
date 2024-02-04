#pragma once

#include <libspdl/core/interface/base.h>

#include <libspdl/core/types.h>

#include <memory>
#include <optional>
#include <string>

struct AVIOContext;
struct AVFormatContext;

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// CustomAdoptor
// Custom adoptor that uses custom AVIOContextPtr to define the I/O
////////////////////////////////////////////////////////////////////////////////
namespace detail {
AVIOContext* get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence));

AVFormatContext* get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options);

void free_av_io_ctx(AVIOContext*);
void free_av_fmt_ctx(AVFormatContext*);
} // namespace detail

template <typename T>
class CustomAdoptor : public SourceAdoptor {
 public:
  class CustomInterface : public DataInterface {
    T obj;
    AVIOContext* io_ctx;
    AVFormatContext* fmt_ctx;

   public:
    CustomInterface(
        const std::string& url,
        const std::optional<std::string>& format,
        const std::optional<OptionDict>& format_options,
        int buffer_size)
        : obj(url),
          io_ctx(
              detail::get_io_ctx(&obj, buffer_size, T::read_packet, T::seek)),
          fmt_ctx(
              detail::get_input_format_ctx(io_ctx, format, format_options)) {}

    ~CustomInterface() {
      detail::free_av_io_ctx(io_ctx);
      detail::free_av_fmt_ctx(fmt_ctx);
    };
    AVFormatContext* get_fmt_ctx() override {
      return fmt_ctx;
    }
  };

 private:
  std::optional<std::string> prefix;
  std::optional<std::string> format;
  std::optional<OptionDict> format_options;
  int buffer_size;

 public:
  CustomAdoptor(
      const std::optional<std::string>& prefix_ = std::nullopt,
      const std::optional<std::string>& format_ = std::nullopt,
      const std::optional<OptionDict>& format_options_ = std::nullopt,
      int buffer_size_ = 8096)
      : prefix(prefix_),
        format(format_),
        format_options(format_options_),
        buffer_size(buffer_size_) {}

  std::unique_ptr<DataInterface> get(const std::string& url) override {
    return std::make_unique<CustomInterface>(
        prefix ? prefix.value() + url : url,
        format,
        format_options,
        buffer_size);
  };
};

} // namespace spdl::core