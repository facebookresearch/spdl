/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/adaptor.h>

#include "libspdl/core/detail/ffmpeg/ctx_utils.h"

#include <memory>

using namespace spdl::core::detail;

namespace spdl::core {
namespace {
class BasicInterface : public DataInterface {
  std::string url;
  detail::AVFormatInputContextPtr fmt_ctx;

 public:
  BasicInterface(std::string url_, const DemuxConfig& dmx_cfg)
      : url(std::move(url_)),
        fmt_ctx(get_input_format_ctx(
            url.c_str(),
            dmx_cfg.format,
            dmx_cfg.format_options)) {}
  ~BasicInterface() override = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
  std::string get_src() const override {
    return url;
  }
};
} // namespace
std::unique_ptr<DataInterface> SourceAdaptor::get_interface(
    std::string_view url,
    const DemuxConfig& dmx_cfg) const {
  return std::unique_ptr<DataInterface>(
      new BasicInterface(std::string{url}, dmx_cfg));
}
} // namespace spdl::core
