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

namespace nb = nanobind;

using int_array = nb::ndarray<nb::device::cpu, nb::c_contig, int64_t>;
using rgb_frames = nb::
    ndarray<uint8_t, nb::shape<-1, -1, -1, 3>, nb::device::cpu, nb::c_contig>;

namespace spdl::core {
namespace {
template <MediaType media_type>
CPUBufferPtr convert(
    const FramesPtr<media_type>&& frames,
    std::shared_ptr<CPUStorage> storage) {
  nb::gil_scoped_release __g;
  return convert_frames(frames.get(), storage);
}

template <MediaType media_type>
std::vector<const spdl::core::Frames<media_type>*> _ref(
    std::vector<FramesPtr<media_type>>& frames) {
  std::vector<const spdl::core::Frames<media_type>*> ret;
  for (auto& frame : frames) {
    ret.push_back(frame.get());
  }
  return ret;
}

template <MediaType media_type>
CPUBufferPtr batch_convert(
    std::vector<FramesPtr<media_type>>&& frames,
    std::shared_ptr<CPUStorage> storage) {
  nb::gil_scoped_release __g;
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

  nb::gil_scoped_release __g; // do not access vals from here.
  auto buf = cpu_buffer(shape, ElemClass::Int, 8, std::move(storage));

  // copy
  int64_t* dst = static_cast<int64_t*>(buf->data());
  memcpy(dst, src, in_size);
  return buf;
}

VideoFramesPtr _convert_rgb_array(
    const rgb_frames& array,
    std::tuple<int, int> frame_rate,
    int pts) {
  // Obtain shape
  std::vector<size_t> shape;
  for (size_t i = 0; i < array.ndim(); ++i) {
    shape.push_back(array.shape(i));
  }
  Rational time_base{std::get<1>(frame_rate), std::get<0>(frame_rate)};
  auto src = array.data();

  nb::gil_scoped_release __g; // do not access vals from here.
  return convert_rgb_array(
      src,
      array.shape(0), // N
      array.shape(1), // H
      array.shape(2), // W
      time_base,
      pts);
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

  ////////////////////////////////////////////////////////////////////////////////
  // Conversion from tensor to AVFrames
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "convert_rgb_array",
      &_convert_rgb_array,
      nb::arg("array"),
      nb::arg("frame_rate"),
      nb::arg("pts"));
}
} // namespace spdl::core
