/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

#include "register_spdl_core_extensions.h"

#include <nanobind/nanobind.h>

namespace {
NB_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  spdl::core::register_types(m);
  spdl::core::register_packets(m);
  spdl::core::register_frames(m);
  spdl::core::register_storage(m);
  spdl::core::register_buffers(m);
  spdl::core::register_tracing(m);
  spdl::core::register_utils(m);
  spdl::core::register_demuxing(m);
  spdl::core::register_decoding(m);
  spdl::core::register_conversion(m);
  spdl::core::register_encoding(m);
  spdl::core::register_filtering(m);
  spdl::core::register_bsf(m);
}
} // namespace
