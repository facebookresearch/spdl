/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/conversion.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "spdl_gil.h"

namespace nb = nanobind;

using int_array = nb::ndarray<nb::device::cpu, nb::c_contig, int64_t>;

namespace spdl::core {
namespace {
template <MediaType media_type>
CPUBufferPtr convert(
    const FFmpegFramesPtr<media_type>&& frames,
    std::shared_ptr<CPUStorage> storage) {
  RELEASE_GIL();
  return convert_frames(frames.get(), storage);
}

template <MediaType media_type>
std::vector<const spdl::core::FFmpegFrames<media_type>*> _ref(
    std::vector<FFmpegFramesPtr<media_type>>& frames) {
  std::vector<const spdl::core::FFmpegFrames<media_type>*> ret;
  for (auto& frame : frames) {
    ret.push_back(frame.get());
  }
  return ret;
}

template <MediaType media_type>
CPUBufferPtr batch_convert(
    std::vector<FFmpegFramesPtr<media_type>>&& frames,
    std::shared_ptr<CPUStorage> storage) {
  RELEASE_GIL();
  return convert_frames(_ref(frames), storage);
}

CPUBufferPtr convert_array(
    int_array vals,
    std::shared_ptr<CPUStorage> storage) {
  auto in_size = vals.nbytes();
  if (in_size == 0) {
    throw std::runtime_error("The array be empty.");
  }
  // Obtain shape
  std::vector<size_t> shape;
  for (size_t i = 0; i < vals.ndim(); ++i) {
    shape.push_back(vals.shape(i));
  }
  auto src = vals.data();

  RELEASE_GIL(); // do not access vals from here.
  auto buf = cpu_buffer(shape, ElemClass::Int, 8, std::move(storage));

  // copy
  int64_t* dst = static_cast<int64_t*>(buf->data());
  memcpy(dst, src, in_size);
  return buf;
}

} // namespace

void register_conversion(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Frame conversion
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "convert_frames",
      &convert<MediaType::Audio>,
      nb::arg("frames"),
      nb::arg("storage") = nullptr);
  m.def(
      "convert_frames",
      &convert<MediaType::Video>,
      nb::arg("frames"),
      nb::arg("storage") = nullptr);
  m.def(
      "convert_frames",
      &convert<MediaType::Image>,
      nb::arg("frames"),
      nb::arg("storage") = nullptr);

  m.def(
      "convert_frames",
      &batch_convert<MediaType::Audio>,
      nb::arg("frames"),
      nb::arg("storage") = nullptr);
  m.def(
      "convert_frames",
      &batch_convert<MediaType::Video>,
      nb::arg("frames"),
      nb::arg("storage") = nullptr);
  m.def(
      "convert_frames",
      &batch_convert<MediaType::Image>,
      nb::arg("frames"),
      nb::arg("storage") = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Conversion from int list
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "convert_array",
      &convert_array,
      nb::arg("vals"),
      nb::arg("storage") = nullptr);
}
} // namespace spdl::core
