#include <libspdl/core/demuxing.h>

#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
namespace {

template <MediaType media_type>
DemuxerPtr<media_type> _make_demuxer(
    const std::string& src,
    const std::optional<DemuxConfig>& dmx_cfg,
    const SourceAdaptorPtr _adaptor) {
  nb::gil_scoped_release g;
  return make_demuxer<media_type>(src, std::move(_adaptor), std::move(dmx_cfg));
}

template <MediaType media_type>
DemuxerPtr<media_type> _make_demuxer_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  auto data_ = std::string_view{data.c_str(), data.size()};
  nb::gil_scoped_release g;
  return make_demuxer<media_type>(data_, std::move(dmx_cfg));
}

template <MediaType media_type>
std::tuple<DemuxerPtr<media_type>, PacketsPtr<media_type>> demuxer_demux(
    DemuxerPtr<media_type> demuxer,
    const std::optional<std::tuple<double, double>>& window) {
  nb::gil_scoped_release g;
  auto packets = demuxer->demux_window(window);
  return {std::move(demuxer), std::move(packets)};
}

template <MediaType media_type>
void drop_demuxer(DemuxerPtr<media_type> t) {
  nb::gil_scoped_release g;
  t.reset(); // Technically not necessary, but it's explicit this way.
}

template <MediaType media_type>
PacketsPtr<media_type> _demux_src(
    const std::string& src,
    const std::optional<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    const SourceAdaptorPtr adaptor) {
  nb::gil_scoped_release g;
  auto demuxer = make_demuxer<media_type>(src, adaptor, dmx_cfg);
  return demuxer->demux_window(timestamps);
}

void zero_clear(nb::bytes data) {
  std::memset((void*)data.c_str(), 0, data.size());
}

template <MediaType media_type>
PacketsPtr<media_type> _demux_bytes(
    nb::bytes data,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool _zero_clear) {
  auto data_ = std::string_view{data.c_str(), data.size()};
  nb::gil_scoped_release g;
  auto demuxer = make_demuxer<media_type>(data_, dmx_cfg);
  auto ret = demuxer->demux_window(timestamp);
  if (_zero_clear) {
    nb::gil_scoped_acquire gg;
    zero_clear(data);
  }
  return ret;
}

ImagePacketsPtr _demux_image(
    const std::string& src,
    const std::optional<DemuxConfig>& dmx_cfg,
    const SourceAdaptorPtr _adaptor) {
  return _demux_src<MediaType::Image>(src, std::nullopt, dmx_cfg, _adaptor);
}

ImagePacketsPtr _demux_image_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg,
    bool _zero_clear) {
  return _demux_bytes<MediaType::Image>(
      data, std::nullopt, dmx_cfg, _zero_clear);
}

} // namespace

void register_demuxing(nb::module_& m) {
  ///////////////////////////////////////////////////////////////////////////////
  // Demuxer
  ///////////////////////////////////////////////////////////////////////////////
  nb::class_<Demuxer<MediaType::Audio>>(m, "AudioDemuxer");
  nb::class_<Demuxer<MediaType::Video>>(m, "VideoDemuxer");

  m.def(
      "_audio_demuxer",
      &_make_demuxer<MediaType::Audio>,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "_audio_demuxer",
      &_make_demuxer_bytes<MediaType::Audio>,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none());

  m.def(
      "_video_demuxer",
      &_make_demuxer<MediaType::Video>,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "_video_demuxer",
      &_make_demuxer_bytes<MediaType::Video>,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none());

  m.def("_demux", &demuxer_demux<MediaType::Audio>);

  m.def("_demux", &demuxer_demux<MediaType::Video>);

  m.def("_drop", &drop_demuxer<MediaType::Audio>);

  m.def("_drop", &drop_demuxer<MediaType::Video>);

  ///////////////////////////////////////////////////////////////////////////////
  // Demux from src (path, URL etc...)
  ///////////////////////////////////////////////////////////////////////////////
  m.def(
      "demux_audio",
      &_demux_src<MediaType::Audio>,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "demux_video",
      &_demux_src<MediaType::Video>,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "demux_image",
      &_demux_image,
      nb::arg("src"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  ///////////////////////////////////////////////////////////////////////////////
  // Demux from byte string
  ///////////////////////////////////////////////////////////////////////////////
  m.def(
      "demux_audio",
      &_demux_bytes<MediaType::Audio>,
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_zero_clear") = false);

  m.def(
      "demux_video",
      &_demux_bytes<MediaType::Video>,
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("_zero_clear") = false);

  m.def(
      "demux_image",
      &_demux_image_bytes,
      nb::arg("data"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("demux_config") = nb::none(),
      nb::arg("_zero_clear") = false);
}
} // namespace spdl::core
