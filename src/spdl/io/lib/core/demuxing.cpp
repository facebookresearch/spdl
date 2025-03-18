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

#include "spdl_gil.h"

namespace nb = nanobind;

using cpu_array = nb::ndarray<const uint8_t, nb::ndim<1>, nb::device::cpu>;

namespace spdl::core {
namespace {

// Wrap Demuxer to
// 1. Release GIL
// 2. Allow explicit deallocation with `drop` method, so that it can be
//    deallocated in a thread other than the main thread.
// 3. Add `zero_clear`, method for testing purpose.

template <MediaType media_type>
struct PyStreamingDemuxer {
  StreamingDemuxerPtr<media_type> demuxer;

  explicit PyStreamingDemuxer(StreamingDemuxerPtr<media_type>&& d)
      : demuxer(std::move(d)) {}

  bool done() {
    RELEASE_GIL();
    return demuxer->done();
  }

  PacketsPtr<media_type> next() {
    RELEASE_GIL();
    return demuxer->next();
  }
};

template <MediaType media_type>
using PyStreamingDemuxerPtr = std::unique_ptr<PyStreamingDemuxer<media_type>>;

struct PyDemuxer {
  DemuxerPtr demuxer;

  std::string_view data;
  bool zero_clear = false;

  explicit PyDemuxer(DemuxerPtr&& demuxer_) : demuxer(std::move(demuxer_)) {}
  PyDemuxer(DemuxerPtr&& demuxer_, std::string_view data_, bool zero_clear_)
      : demuxer(std::move(demuxer_)), data(data_), zero_clear(zero_clear_) {}

  bool has_audio() {
    RELEASE_GIL();
    return demuxer->has_audio();
  }

  template <MediaType media_type>
  Codec<media_type> get_default_codec() const {
    return demuxer->get_default_codec<media_type>();
  }

  template <MediaType media_type>
  PacketsPtr<media_type> demux(
      const std::optional<std::tuple<double, double>>& window,
      const std::optional<std::string>& bsf) {
    RELEASE_GIL();
    return demuxer->demux_window<media_type>(window, bsf);
  }

  PacketsPtr<MediaType::Image> demux_image(
      const std::optional<std::string>& bsf) {
    RELEASE_GIL();
    return demuxer->demux_window<MediaType::Image>(std::nullopt, bsf);
  }

  template <MediaType media_type>
  PyStreamingDemuxerPtr<media_type> streaming_demux(
      int num_packets,
      const std::optional<std::string>& bsf) {
    RELEASE_GIL();
    return std::make_unique<PyStreamingDemuxer<media_type>>(
        demuxer->stream_demux<media_type>(num_packets, bsf));
  }

  void _drop() {
    RELEASE_GIL();
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
  RELEASE_GIL();
  return std::make_unique<PyDemuxer>(
      make_demuxer(src, std::move(_adaptor), dmx_cfg));
}

PyDemuxerPtr _make_demuxer_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool zero_clear = false) {
  auto data_ = std::string_view{data.c_str(), data.size()};
  RELEASE_GIL();
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
  RELEASE_GIL();
  return std::make_unique<PyDemuxer>(
      make_demuxer(data_, dmx_cfg), data_, zero_clear);
}

} // namespace

void register_demuxing(nb::module_& m) {
  ///////////////////////////////////////////////////////////////////////////////
  // Demuxer
  ///////////////////////////////////////////////////////////////////////////////
  nb::class_<PyStreamingDemuxer<MediaType::Video>>(m, "StreamingVideoDemuxer")
      .def("done", &PyStreamingDemuxer<MediaType::Video>::done)
      .def("next", &PyStreamingDemuxer<MediaType::Video>::next);

  nb::class_<AudioCodec>(m, "AudioCodec")
      .def_prop_ro("name", &AudioCodec::get_name);
  nb::class_<VideoCodec>(m, "VideoCodec")
      .def_prop_ro("name", &VideoCodec::get_name);
  nb::class_<ImageCodec>(m, "ImageCodec")
      .def_prop_ro("name", &ImageCodec::get_name);

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
      .def_prop_ro(
          "audio_codec", &PyDemuxer::get_default_codec<MediaType::Audio>)
      .def_prop_ro(
          "video_codec", &PyDemuxer::get_default_codec<MediaType::Video>)
      .def_prop_ro(
          "image_codec", &PyDemuxer::get_default_codec<MediaType::Image>)
      .def(
          "streaming_demux_video",
          &PyDemuxer::streaming_demux<MediaType::Video>,
          nb::arg("num_packets"),
          nb::arg("bsf") = nb::none())
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

  // TEMP: as a preprocessing for NVDEC decoding
  m.def("_apply_bsf", &apply_bsf, nb::arg("packets"));
}
} // namespace spdl::core
