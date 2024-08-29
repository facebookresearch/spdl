/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#ifdef SPDL_LOG_API_USAGE
#include <c10/util/Logging.h>
#include <fmt/core.h>
#endif

namespace nb = nanobind;

namespace spdl::core {

void register_utils(nb::module_& m) {
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("register_avdevices", &register_avdevices);
  m.def("get_ffmpeg_filters", &get_ffmpeg_filters);

  m.def("is_cuda_available", []() {
    nb::gil_scoped_release g;
    return is_cuda_available();
  });
  m.def("is_nvcodec_available", []() {
    nb::gil_scoped_release g;
    return is_nvcodec_available();
  });

  m.def("init_glog", [](char const* name) {
    nb::gil_scoped_release g;
    init_glog(name);
  });

  m.def("log_api_usage", [](const std::string& event) {
#ifndef SPDL_LOG_API_USAGE
    return;
#else
  // Note:
  // We cannot use C10_LOG_API_USAGE_ONCE macro with dynamically created event
  // name.
  //
  // This is because, at compile time, the macro is replaced with
  // initialization of static variable, and perform logging as a side effect
  // of the initialization. (therefore it's called only once)
  //
  // https://github.com/pytorch/pytorch/blob/144639797a44bac5de60700ecc024f00f32a8515/c10/util/Logging.h#L290-L304
  //
  // PyTorch solves this by caching the event name.
  // This implementation relies on the fact that Python GIL will ensure the
  // exclusive access.
  // https://github.com/pytorch/pytorch/blob/784a6ec5a30bd2d1831cb4f78183ad51696794e5/torch/csrc/Module.cpp#L1763
  // https://github.com/pytorch/pytorch/blob/784a6ec5a30bd2d1831cb4f78183ad51696794e5/torch/csrc/Module.cpp#L1579-L1587
  //
  // However, as we enter the no-GIL era, and SPDL is all about multi-threading,
  // replying on GIL does not feel secure enough. Since this function is not a
  // public API in SPDL, and we use it for only handful functions, so we hard
  // code the supported patterns.

#define LOG_API(name)             \
  if (event == name) {            \
    C10_LOG_API_USAGE_ONCE(name); \
    return;                       \
  }

  LOG_API("spdl");
  LOG_API("spdl.dataloader.Pipeline");
  LOG_API("spdl.io.demux_audio");
  LOG_API("spdl.io.demux_video");
  LOG_API("spdl.io.demux_image");

#ifdef DEBUG
  throw std::runtime_error(fmt::format("[INTERNAL] The event name {} is not registered.", name));
#endif
#endif
  });
}

} // namespace spdl::core
