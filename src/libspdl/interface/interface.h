#pragma once

#include <libspdl/common.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

struct AVFormatContext;

namespace spdl::interface {

// DataProvider serves two purposes:
// 1. It encapsulates the AVFormatContext, AVIOContext and other
//    objects so that their lifetimes are aligned.
// 2. Abstract away the difference between the interfaces. Each
//    interface can require additional/specific implementation to
//    talk to the actual data source. Such things are distilled to
//    merely AVFormatContext*.
struct DataProvider {
  virtual ~DataProvider() = default;
  virtual AVFormatContext* get_fmt_ctx() = 0;
};

std::unique_ptr<DataProvider> get_data_provider(
    const std::string& url,
    const std::optional<std::string>& format = std::nullopt,
    const std::optional<OptionDict>& format_options = std::nullopt,
    int buffer_size = 8096);

} // namespace spdl::interface
