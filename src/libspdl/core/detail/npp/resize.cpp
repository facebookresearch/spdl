/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/npp/resize.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/npp/utils.h"
#include "libspdl/core/detail/nvjpeg/utils.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

#include <nppi.h>

#define RESIZE_FUNC         \
  NppStatus (*resize_func)( \
      const Npp8u*,         \
      int,                  \
      NppiSize,             \
      NppiRect,             \
      Npp8u*,               \
      int,                  \
      NppiSize,             \
      NppiRect,             \
      int,                  \
      NppStreamContext)

namespace spdl::core::detail {
namespace {

template <RESIZE_FUNC>
void resize(
    const nvjpegImage_t& src,
    const NppiSize& src_size,
    const NppiRect& src_roi,
    nvjpegImage_t& dst,
    const NppiSize& dst_size,
    const NppiRect& dst_roi,
    NppStreamContext& stream,
    int index = 0) {
  TRACE_EVENT("decoding", "nppiResize");
  CHECK_NPP(
      resize_func(
          src.channel[index],
          src.pitch[index],
          src_size,
          src_roi,
          dst.channel[index],
          dst.pitch[index],
          dst_size,
          src_roi,
          NPPI_INTER_LANCZOS,
          stream),
      "Failed to resize the image.");
}

} // namespace

void resize_npp(
    nvjpegOutputFormat_t fmt,
    nvjpegImage_t src,
    int src_width,
    int src_height,
    nvjpegImage_t dst,
    int dst_width,
    int dst_height) {
#ifndef SPDL_USE_NPPI
  SPDL_FAIL(
      "Image resizing while decoding with NVJPEG reqreuires SPDL to be compiled with NPPI support.");
#else
  NppStreamContext stream;
  stream.hStream = nullptr; // default stream

  NppiSize src_size{.width = src_width, .height = src_height};
  NppiSize dst_size{.width = dst_width, .height = dst_height};

  // TODO: support ROI
  NppiRect src_roi{.x = 0, .y = 0, .width = src_width, .height = src_height};
  NppiRect dst_roi{.x = 0, .y = 0, .width = dst_width, .height = dst_height};

  switch (fmt) {
    case NVJPEG_OUTPUT_RGBI:
      [[fallthrough]];
    case NVJPEG_OUTPUT_BGRI:
      resize<nppiResize_8u_C3R_Ctx>(
          src, src_size, src_roi, dst, dst_size, dst_roi, stream);
      break;
    case NVJPEG_OUTPUT_Y:
      resize<nppiResize_8u_C1R_Ctx>(
          src, src_size, src_roi, dst, dst_size, dst_roi, stream);
      break;
    case NVJPEG_OUTPUT_RGB:
      [[fallthrough]];
    case NVJPEG_OUTPUT_BGR:
      resize<nppiResize_8u_C1R_Ctx>(
          src, src_size, src_roi, dst, dst_size, dst_roi, stream, 0);
      resize<nppiResize_8u_C1R_Ctx>(
          src, src_size, src_roi, dst, dst_size, dst_roi, stream, 1);
      resize<nppiResize_8u_C1R_Ctx>(
          src, src_size, src_roi, dst, dst_size, dst_roi, stream, 2);
      break;
    default:
      // It should be already handled by `get_nvjpeg_output_format`
      SPDL_FAIL_INTERNAL(
          fmt::format("Unexpected output format: {}", to_string(fmt)));
  }
#endif
}

} // namespace spdl::core::detail
