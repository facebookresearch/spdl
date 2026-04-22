/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <nanobind/nanobind.h>
#include <cstddef>
#include <string_view>

namespace spdl::detail {

inline std::string_view memoryview_to_sv(const nanobind::memoryview& mv) {
  Py_buffer* buf = PyMemoryView_GET_BUFFER(mv.ptr());
  return {static_cast<const char*>(buf->buf), static_cast<size_t>(buf->len)};
}

} // namespace spdl::detail
