/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <optional>
#include <string>

namespace spdl::core::detail {

AVIOContextPtr get_io_ctx(
    void* opaque,
    int buffer_size,
    int (*read_packet)(void* opaque, uint8_t* buf, int buf_size),
    int64_t (*seek)(void* opaque, int64_t offset, int whence));

AVFormatInputContextPtr get_input_format_ctx(
    const std::string& url,
    const std::optional<std::string>& format = std::nullopt,
    const std::optional<OptionDict>& format_options = std::nullopt);

AVFormatInputContextPtr get_input_format_ctx(
    AVIOContext* io_ctx,
    const std::optional<std::string>& format,
    const std::optional<OptionDict>& format_options);

AVCodecContextPtr get_decode_codec_ctx_ptr(
    const AVCodecParameters* params,
    Rational pkt_timebase,
    const std::optional<std::string>& decoder = std::nullopt,
    const std::optional<OptionDict>& decoder_options = std::nullopt);

///////////////////////////////////////////////////////////////////////////////
// Encoding
///////////////////////////////////////////////////////////////////////////////

AVFormatOutputContextPtr get_output_format_ctx(
    const std::string& url,
    const std::optional<std::string>& format = std::nullopt);

void open_codec_for_encode(
    AVCodecContext* codec_ctx,
    const std::optional<OptionDict>& option);

void open_format(
    AVFormatContext* format_ctx,
    const std::optional<OptionDict>& option = std::nullopt);

} // namespace spdl::core::detail
