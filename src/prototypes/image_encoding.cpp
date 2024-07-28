/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/conversion.h>
#include <libspdl/core/decoding.h>
#include <libspdl/core/demuxing.h>
#include <libspdl/core/encoding.h>
#include <libspdl/core/types.h>

#include <fmt/core.h>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
}

void reenc(
    const std::string& input,
    const std::string& output,
    const std::string& pix_fmt,
    const std::string& filter_desc) {
  using namespace spdl::core;

  spdl::core::EncodeConfig enc_cfg{.format = pix_fmt};

  auto demuxer = make_demuxer(input);
  auto packets = demuxer->demux_window<MediaType::Image>();
  auto frames =
      decode_packets_ffmpeg(std::move(packets), std::nullopt, filter_desc);
  auto buffer = convert_frames(frames.get());
  if (pix_fmt == "gray16be" || "gray8") {
    buffer->shape.erase(buffer->shape.begin());
  }
  encode_image(
      output, buffer->data(), buffer->shape, buffer->depth, pix_fmt, enc_cfg);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fmt::print("Usage: {} <input> <output>\n", argv[0]);
    return 1;
  }
  const std::string src = argv[1];
  const std::string dst = argv[2];
  const std::string pix_fmt = "gray16be";

  const std::string filter_desc = fmt::format("format=pix_fmts={}", pix_fmt);
  reenc(src, dst, pix_fmt, filter_desc);
}
