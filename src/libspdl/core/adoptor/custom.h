#pragma once

#include <libspdl/core/adoptor/base.h>

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
struct CustomAdoptor : public SourceAdoptor {
  class CustomInterface : public DataInterface {
    T obj;
    AVIOContext* io_ctx;
    AVFormatContext* fmt_ctx;

   public:
    CustomInterface(const std::string& url, const IOConfig& io_cfg)
        : obj(url),
          io_ctx(detail::get_io_ctx(
              &obj,
              io_cfg.buffer_size,
              T::read_packet,
              T::seek)),
          fmt_ctx(detail::get_input_format_ctx(
              io_ctx,
              io_cfg.format,
              io_cfg.format_options)) {}

    ~CustomInterface() {
      detail::free_av_io_ctx(io_ctx);
      detail::free_av_fmt_ctx(fmt_ctx);
    };
    AVFormatContext* get_fmt_ctx() override {
      return fmt_ctx;
    }
  };

  const std::optional<std::string> prefix;

  CustomAdoptor(const std::optional<std::string>& prefix_ = std::nullopt)
      : prefix(prefix_) {}

  void* get(const std::string& url, const IOConfig& io_cfg) const override {
    return new CustomInterface(prefix ? prefix.value() + url : url, io_cfg);
  };
};

} // namespace spdl::core
