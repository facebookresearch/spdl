/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>

#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using cpu_array = nb::ndarray<const uint8_t, nb::ndim<1>, nb::device::cpu>;

namespace spdl::core {
namespace {

// Wrap Demuxer to
// 1. Release GIL
// 2. Allow explicit deallocation with `drop` method, so that it can be
//    deallocated in a thread other than the main thread.
// 3. Add `zero_clear`, method for testing purpose.
struct PyDemuxer {
  DemuxerPtr demuxer;

  std::string_view data;
  bool zero_clear = false;

  explicit PyDemuxer(DemuxerPtr&& demuxer_) : demuxer(std::move(demuxer_)) {}
  PyDemuxer(DemuxerPtr&& demuxer_, std::string_view data_, bool zero_clear_)
      : demuxer(std::move(demuxer_)), data(data_), zero_clear(zero_clear_) {}

  bool has_audio() {
    nb::gil_scoped_release g;
    return demuxer->has_audio();
  }

  template <MediaType media_type>
  PacketsPtr<media_type> demux(
      const std::optional<std::tuple<double, double>>& window,
      const std::optional<std::string>& bsf) {
    nb::gil_scoped_release g;
    return demuxer->demux_window<media_type>(window, bsf);
  }

  PacketsPtr<MediaType::Image> demux_image(
      const std::optional<std::string>& bsf) {
    nb::gil_scoped_release g;
    return demuxer->demux_window<MediaType::Image>(std::nullopt, bsf);
  }

  void _drop() {
    nb::gil_scoped_release g;
    if (zero_clear) {
      std::memset((void*)data.data(), 0, data.size());
    }
    // Explicitly release
    demuxer.reset();
  }
};

using PyDemuxerPtr = std::unique_ptr<PyDemuxer>;

PyDemuxerPtr _make_demuxer(
    const std::string& src,
    const std::optional<DemuxConfig>& dmx_cfg,
    SourceAdaptorPtr _adaptor) {
  nb::gil_scoped_release g;
  return std::make_unique<PyDemuxer>(
      make_demuxer(src, std::move(_adaptor), dmx_cfg));
}

PyDemuxerPtr _make_demuxer_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool zero_clear = false) {
  auto data_ = std::string_view{data.c_str(), data.size()};
  nb::gil_scoped_release g;
  return std::make_unique<PyDemuxer>(
      make_demuxer(data_, dmx_cfg), data_, zero_clear);
}

PyDemuxerPtr _make_demuxer_array(
    cpu_array data,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool zero_clear = false) {
  auto ptr = reinterpret_cast<const char*>(data.data());
  // Note size() returns the number of elements, not the size in bytes.
  auto data_ = std::string_view{ptr, data.size()};
  nb::gil_scoped_release g;
  return std::make_unique<PyDemuxer>(
      make_demuxer(data_, dmx_cfg), data_, zero_clear);
}

} // namespace

void register_demuxing(nb::module_& m) {
  ///////////////////////////////////////////////////////////////////////////////
  // Demuxer
  ///////////////////////////////////////////////////////////////////////////////
  nb::class_<PyDemuxer>(m, "Demuxer")
      .def(
          "demux_audio",
          &PyDemuxer::demux<MediaType::Audio>,
          nb::arg("window") = nb::none(),
          nb::arg("bsf") = nb::none())
      .def(
          "demux_video",
          &PyDemuxer::demux<MediaType::Video>,
          nb::arg("window") = nb::none(),
          nb::arg("bsf") = nb::none())
      .def("demux_image", &PyDemuxer::demux_image, nb::arg("bsf") = nb::none())
      .def("has_audio", &PyDemuxer::has_audio)
      .def("_drop", &PyDemuxer::_drop);

  m.def(
      "_demuxer",
      &_make_demuxer,
      nb::arg("src"),
      nb::kw_only(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "_demuxer",
      &_make_demuxer_bytes,
      nb::arg("src"),
      nb::kw_only(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_zero_clear") = false);
  m.def(
      "_demuxer",
      &_make_demuxer_array,
      nb::arg("src"),
      nb::kw_only(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_zero_clear") = false);
}
} // namespace spdl::core
