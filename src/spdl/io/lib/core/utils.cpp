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
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "spdl_gil.h"

namespace nb = nanobind;

namespace spdl::core {

void register_utils(nb::module_& m) {
  m.def("get_ffmpeg_log_level", &get_ffmpeg_log_level);
  m.def("set_ffmpeg_log_level", &set_ffmpeg_log_level);
  m.def("register_avdevices", &register_avdevices);
  m.def("get_ffmpeg_filters", &get_ffmpeg_filters);
  m.def("get_ffmpeg_versions", &get_ffmpeg_versions);

  m.def("is_cuda_available", []() {
    RELEASE_GIL();
    return is_cuda_available();
  });
  m.def("is_nvcodec_available", []() {
    RELEASE_GIL();
    return is_nvcodec_available();
  });
  m.def("is_nvjpeg_available", []() {
    RELEASE_GIL();
    return is_nvjpeg_available();
  });

  m.def("init_glog", [](char const* name) {
    RELEASE_GIL();
    init_glog(name);
  });
}

} // namespace spdl::core
