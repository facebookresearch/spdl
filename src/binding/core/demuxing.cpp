#include <libspdl/core/demuxing.h>

#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#ifdef SPDL_LOG_API_USAGE
#include <c10/util/Logging.h>

#define _USAGE_LOGGING(name)                      \
  if constexpr (media_type == MediaType::Audio) { \
    C10_LOG_API_USAGE_ONCE(name "_audio");        \
  }                                               \
  if constexpr (media_type == MediaType::Video) { \
    C10_LOG_API_USAGE_ONCE(name "_video");        \
  }                                               \
  if constexpr (media_type == MediaType::Image) { \
    C10_LOG_API_USAGE_ONCE(name "_image");        \
  }

#else
#define _USAGE_LOGGING(...)
#endif

namespace nb = nanobind;

namespace spdl::core {
namespace {

DemuxerPtr _make_demuxer(
    const std::string& src,
    const std::optional<DemuxConfig>& dmx_cfg,
    SourceAdaptorPtr _adaptor) {
  nb::gil_scoped_release g;
  return make_demuxer(src, std::move(_adaptor), dmx_cfg);
}

DemuxerPtr _make_demuxer_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg) {
  auto data_ = std::string_view{data.c_str(), data.size()};
  nb::gil_scoped_release g;
  return make_demuxer(data_, dmx_cfg);
}

template <MediaType media_type>
std::tuple<DemuxerPtr, PacketsPtr<media_type>> _demuxer_demux(
    DemuxerPtr demuxer,
    const std::optional<std::tuple<double, double>>& window,
    const std::optional<std::string>& bsf) {
  _USAGE_LOGGING("spdl.core.demux_window");
  nb::gil_scoped_release g;
  auto packets = demuxer->demux_window<media_type>(window, bsf);
  return {std::move(demuxer), std::move(packets)};
}

void drop_demuxer(DemuxerPtr t) {
  nb::gil_scoped_release g;
  t.reset(); // Technically not necessary, but it's explicit this way.
}

template <MediaType media_type>
PacketsPtr<media_type> _demux_src(
    const std::string& src,
    const std::optional<std::tuple<double, double>>& timestamps,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& bsf,
    const SourceAdaptorPtr adaptor) {
  _USAGE_LOGGING("spdl.core.demux_src");
  nb::gil_scoped_release g;
  auto demuxer = make_demuxer(src, adaptor, dmx_cfg);
  return demuxer->demux_window<media_type>(timestamps, bsf);
}

void zero_clear(nb::bytes data) {
  std::memset((void*)data.c_str(), 0, data.size());
}

template <MediaType media_type>
PacketsPtr<media_type> _demux_bytes(
    nb::bytes data,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& bsf,
    bool _zero_clear) {
  _USAGE_LOGGING("spdl.core.demux_bytes");
  auto data_ = std::string_view{data.c_str(), data.size()};
  nb::gil_scoped_release g;
  auto demuxer = make_demuxer(data_, dmx_cfg);
  auto ret = demuxer->demux_window<media_type>(timestamp, bsf);
  if (_zero_clear) {
    nb::gil_scoped_acquire gg;
    zero_clear(data);
  }
  return ret;
}

ImagePacketsPtr _demux_image(
    const std::string& src,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& bsf,
    const SourceAdaptorPtr _adaptor) {
  return _demux_src<MediaType::Image>(
      src, std::nullopt, dmx_cfg, bsf, _adaptor);
}

ImagePacketsPtr _demux_image_bytes(
    nb::bytes data,
    const std::optional<DemuxConfig>& dmx_cfg,
    const std::optional<std::string>& bsf,
    bool _zero_clear) {
  return _demux_bytes<MediaType::Image>(
      data, std::nullopt, dmx_cfg, bsf, _zero_clear);
}

} // namespace

void register_demuxing(nb::module_& m) {
  ///////////////////////////////////////////////////////////////////////////////
  // Demuxer
  ///////////////////////////////////////////////////////////////////////////////
  nb::class_<Demuxer>(m, "Demuxer");

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
      nb::arg("demux_config") = nb::none());

  m.def(
      "_demux_audio",
      &_demuxer_demux<MediaType::Audio>,
      nb::arg("demuxer"),
      nb::arg("window") = nb::none(),
      nb::arg("bsf") = nb::none());
  m.def(
      "_demux_video",
      &_demuxer_demux<MediaType::Video>,
      nb::arg("demuxer"),
      nb::arg("window") = nb::none(),
      nb::arg("bsf") = nb::none());

  m.def("_drop", &drop_demuxer);

  ///////////////////////////////////////////////////////////////////////////////
  // Demux from src (path, URL etc...)
  ///////////////////////////////////////////////////////////////////////////////
  m.def(
      "demux_audio",
      &_demux_src<MediaType::Audio>,
      nb::arg("src"),
      nb::kw_only(),
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("bsf") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "demux_video",
      &_demux_src<MediaType::Video>,
      nb::arg("src"),
      nb::kw_only(),
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("bsf") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  m.def(
      "demux_image",
      &_demux_image,
      nb::arg("src"),
      nb::kw_only(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("bsf") = nb::none(),
      nb::arg("_adaptor") = nullptr);

  ///////////////////////////////////////////////////////////////////////////////
  // Demux from byte string
  ///////////////////////////////////////////////////////////////////////////////
  m.def(
      "demux_audio",
      &_demux_bytes<MediaType::Audio>,
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("bsf") = nb::none(),
      nb::arg("_zero_clear") = false);

  m.def(
      "demux_video",
      &_demux_bytes<MediaType::Video>,
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("timestamp") = nb::none(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("bsf") = nb::none(),
      nb::arg("_zero_clear") = false);

  m.def(
      "demux_image",
      &_demux_image_bytes,
      nb::arg("data"),
      nb::kw_only(),
      nb::arg("demux_config") = nb::none(),
      nb::arg("bsf") = nb::none(),
      nb::arg("_zero_clear") = false);
}
} // namespace spdl::core
