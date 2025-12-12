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

extern "C" {
#include <libavutil/mem.h>
}

using namespace spdl::core::detail;

namespace spdl::core {
namespace {
class BasicInterface : public DataInterface {
  detail::AVFormatInputContextPtr fmt_ctx;

 public:
  BasicInterface(
      const std::string& url,
      const DemuxConfig& dmx_cfg,
      const std::optional<std::string>& name)
      : fmt_ctx(get_input_format_ctx(
            url.c_str(),
            dmx_cfg.format,
            dmx_cfg.format_options)) {
    if (name) {
      const std::string& nm = *name;
      // Free existing URL if present before overwriting
      if (fmt_ctx->url) {
        av_freep(&fmt_ctx->url);
      }
      fmt_ctx->url = av_strdup(nm.c_str());
    }
  }
  ~BasicInterface() override = default;
  AVFormatContext* get_fmt_ctx() override {
    return fmt_ctx.get();
  }
};
} // namespace

DataInterfacePtr SourceAdaptor::get_interface(
    std::string_view url,
    const DemuxConfig& dmx_cfg,
    const std::optional<std::string>& name) const {
  return DataInterfacePtr(new BasicInterface(std::string{url}, dmx_cfg, name));
}
} // namespace spdl::core
