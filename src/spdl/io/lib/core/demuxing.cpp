/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/demuxing.h>

#include <fmt/core.h>

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
#include <nanobind/stl/variant.h>
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

template <MediaType media>
struct PyStreamingDemuxer {
  StreamingDemuxerPtr demuxer;

  explicit PyStreamingDemuxer(StreamingDemuxerPtr&& d)
      : demuxer(std::move(d)) {}

  bool done() {
    nb::gil_scoped_release __g;
    return demuxer->done();
  }

  AnyPackets next() {
    nb::gil_scoped_release __g;
    return demuxer->next();
  }
};

template <MediaType media>
using PyStreamingDemuxerPtr = std::unique_ptr<PyStreamingDemuxer<media>>;

struct PyDemuxer {
  DemuxerPtr demuxer;

  std::string_view data;
  bool zero_clear = false;

  explicit PyDemuxer(DemuxerPtr&& demuxer_) : demuxer(std::move(demuxer_)) {}
  PyDemuxer(DemuxerPtr&& demuxer_, std::string_view data_, bool zero_clear_)
      : demuxer(std::move(demuxer_)), data(data_), zero_clear(zero_clear_) {}

  bool has_audio() {
    nb::gil_scoped_release __g;
    return demuxer->has_audio();
  }

  template <MediaType media>
  Codec<media> get_default_codec() const {
    return demuxer->get_default_codec<media>();
  }

  template <MediaType media>
  int get_default_stream_index() const {
    return demuxer->get_default_stream_index<media>();
  }

  template <MediaType media>
  PacketsPtr<media> demux(
      const std::optional<std::tuple<double, double>>& window,
      const std::optional<std::string>& bsf) {
    nb::gil_scoped_release __g;
    return demuxer->demux_window<media>(window, bsf);
  }

  PacketsPtr<MediaType::Image> demux_image(
      const std::optional<std::string>& bsf) {
    nb::gil_scoped_release __g;
    return demuxer->demux_window<MediaType::Image>(std::nullopt, bsf);
  }

  template <MediaType media>
  PyStreamingDemuxerPtr<media> streaming_demux(
      int num_packets,
      const std::optional<std::string>& bsf) {
    nb::gil_scoped_release __g;
    return std::make_unique<PyStreamingDemuxer<media>>(
        demuxer->stream_demux(num_packets, bsf));
  }

  void _drop() {
    nb::gil_scoped_release __g;
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
  nb::gil_scoped_release __g;
  return std::make_unique<PyDemuxer>(
      make_demuxer(src, std::move(_adaptor), dmx_cfg));
}

PyDemuxerPtr _make_demuxer_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool zero_clear = false) {
  auto data_ = std::string_view{data.c_str(), data.size()};
  nb::gil_scoped_release __g;
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
  nb::gil_scoped_release __g;
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
      .def_prop_ro(
          "name",
          &AudioCodec::get_name,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "sample_rate",
          &AudioCodec::get_sample_rate,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "num_channels",
          &AudioCodec::get_num_channels,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "sample_fmt",
          &AudioCodec::get_sample_fmt,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "time_base",
          [](VideoCodec& self) -> std::tuple<int, int> {
            auto base = self.get_time_base();
            return {base.num, base.den};
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def("__repr__", [](const AudioCodec& self) {
        return fmt::format(
            "AudioCodec<name={}, sample_rate={}, num_channels={}, sample_fmt={}>",
            self.get_name(),
            self.get_sample_rate(),
            self.get_num_channels(),
            self.get_sample_fmt());
      });
  nb::class_<VideoCodec>(m, "VideoCodec")
      .def_prop_ro(
          "name",
          &VideoCodec::get_name,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          &VideoCodec::get_width,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          &VideoCodec::get_height,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "pix_fmt",
          &VideoCodec::get_pix_fmt,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "frame_rate",
          [](VideoCodec& self) -> std::tuple<int, int> {
            auto rate = self.get_frame_rate();
            return {rate.num, rate.den};
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "time_base",
          [](VideoCodec& self) -> std::tuple<int, int> {
            auto base = self.get_time_base();
            return {base.num, base.den};
          },
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<ImageCodec>(m, "ImageCodec")
      .def_prop_ro(
          "name",
          &ImageCodec::get_name,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "width",
          &ImageCodec::get_width,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "height",
          &ImageCodec::get_height,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "pix_fmt",
          &ImageCodec::get_pix_fmt,
          nb::call_guard<nb::gil_scoped_release>());

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
          "audio_stream_index",
          &PyDemuxer::get_default_stream_index<MediaType::Audio>)
      .def_prop_ro(
          "video_stream_index",
          &PyDemuxer::get_default_stream_index<MediaType::Video>)
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
}
} // namespace spdl::core
