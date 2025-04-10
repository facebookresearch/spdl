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

// NOTE: Do not use `nb::c_contig`
//
// `nb::c_contig` makes nanobind ensures the memory contiguousness by making
// an intermediate copy, which is released soon after.
//
// This is problematic for our usecase.
// - We do not want to have redundant data copy for performance reason.
// - The Frames object we create refers to the original memory region, so
//   we need a way to keep the reference to the object that hold the data.
//   An intermediate copy makes this impossible.
//
// So instead of using `nb::c_contig`, we pass around stride and check
// the contiguousness by ourselves.
using audio_array = nb::ndarray<nb::shape<-1, -1>, nb::device::cpu>;
using video_array = nb::ndarray<nb::device::cpu>;

namespace spdl::core {
namespace {
template <MediaType media>
CPUBufferPtr convert(
    const FramesPtr<media>&& frames,
    std::shared_ptr<CPUStorage> storage) {
  nb::gil_scoped_release __g;
  return convert_frames(frames.get(), storage);
}

template <MediaType media>
std::vector<const spdl::core::Frames<media>*> _ref(
    std::vector<FramesPtr<media>>& frames) {
  std::vector<const spdl::core::Frames<media>*> ret;
  for (auto& frame : frames) {
    ret.push_back(frame.get());
  }
  return ret;
}

template <MediaType media>
CPUBufferPtr batch_convert(
    std::vector<FramesPtr<media>>&& frames,
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
      "create_reference_audio_frame",
      [](const audio_array& array,
         const std::string& sample_fmt,
         int sample_rate,
         int64_t pts) -> AudioFramesPtr {
        assert(array.ndim() == 2); // Guaranteed by nanobind
        // NOTE: nanobind's stride is element count. Not bytes
        std::array<int64_t, 2> stride{array.stride(0), array.stride(1)};
        std::array<size_t, 2> shape{array.shape(0), array.shape(1)};
        auto src = array.data();
        auto bits = array.dtype().bits;

        nb::gil_scoped_release __g; // do not access array from here.
        return create_reference_audio_frame(
            sample_fmt, src, bits, shape, stride, sample_rate, pts);
      },
      nb::arg("array"),
      nb::kw_only(),
      nb::arg("sample_fmt"),
      nb::arg("sample_rate"),
      nb::arg("pts"));

  m.def(
      "create_reference_video_frame",
      [](const video_array& array,
         const std::string& pix_fmt,
         const std::tuple<int, int>& frame_rate,
         int64_t pts) -> VideoFramesPtr {
        auto ndim = array.ndim();
        if (!(ndim == 3 || ndim == 4)) {
          throw std::runtime_error("The input array must be 3D or 4D.");
        }
        auto src = array.data();
        // Obtain shape
        std::vector<size_t> shape;
        std::vector<int64_t> stride;
        for (size_t i = 0; i < array.ndim(); ++i) {
          shape.push_back(array.shape(i));
          stride.push_back(array.stride(i));
        }
        auto bits = array.dtype().bits;

        nb::gil_scoped_release __g; // do not access array from here.
        return create_reference_video_frame(
            pix_fmt,
            src,
            bits,
            shape,
            stride,
            // Flipping frame rate to time base
            Rational{std::get<1>(frame_rate), std::get<0>(frame_rate)},
            pts);
      },
      nb::arg("array"),
      nb::kw_only(),
      nb::arg("pix_fmt"),
      nb::arg("frame_rate"),
      nb::arg("pts"));
}
} // namespace spdl::core
