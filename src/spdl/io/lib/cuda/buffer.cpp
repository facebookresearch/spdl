/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/core/buffer.h>
#include <libspdl/cuda/buffer.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::cuda {
using namespace spdl::core;
////////////////////////////////////////////////////////////////////////////////
// Array interface supplements
////////////////////////////////////////////////////////////////////////////////
#ifdef SPDL_USE_CUDA
namespace {
std::string get_typestr(const ElemClass elem_class, size_t depth) {
  const auto key = [&]() {
    switch (elem_class) {
      case ElemClass::UInt:
        return "u";
      case ElemClass::Int:
        return "i";
      case ElemClass::Float:
        return "f";
      default:
        throw std::runtime_error(
            fmt::format("Unsupported class {}.", int(elem_class)));
    }
  }();
  return fmt::format("|{}{}", key, depth);
}

nb::dict get_cuda_array_interface(CUDABuffer& b) {
  auto typestr = get_typestr(b.elem_class, b.depth);
  nb::dict ret;
  ret["version"] = 2;
  ret["shape"] = nb::tuple(nb::cast(b.shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = nb::none();
  ret["stream"] = b.get_cuda_stream();
  return ret;
}
} // namespace
#endif

// To support -Wunused-parameter
#ifdef SPDL_USE_CUDA
#define _V(var_name) var_name
#define _I(impl) impl
#else
#define _V(var_name)
#define _I(impl) \
  { throw std::runtime_error("SPDL is not compiled with CUDA support."); }
#endif

void register_buffers(nb::module_& m) {
  nb::class_<CUDABuffer>(m, "CUDABuffer")
      .def_prop_ro("__cuda_array_interface__", [](CUDABuffer & _V(self)) _I({
                     return get_cuda_array_interface(self);
                   }))
      .def_prop_ro(
          "device_index",
          [](CUDABuffer & _V(self)) _I({ return self.device_index; }),
          nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::cuda
