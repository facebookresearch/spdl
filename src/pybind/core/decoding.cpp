#include <libspdl/core/decoding.h>

#include <libspdl/core/utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
std::optional<std::tuple<int, int>> get_frame_rate(
    const py::object& frame_rate) {
  if (frame_rate.is(py::none())) {
    return std::nullopt;
  }
  py::object Fraction = py::module_::import("fractions").attr("Fraction");
  py::object r = Fraction(frame_rate);
  return {{r.attr("numerator").cast<int>(), r.attr("denominator").cast<int>()}};
}
} // namespace

void register_pybind(py::module& m) {
  auto _MultipleDecodingResult = py::class_<MultipleDecodingResult>(
      m, "MultipleDecodingResult", py::module_local());

  auto _SingleDecodingResult = py::class_<SingleDecodingResult>(
      m, "SingleDecodingResult", py::module_local());

  _MultipleDecodingResult.def(
      "get", &MultipleDecodingResult::get, py::arg("strict") = true);

  _SingleDecodingResult.def("get", &SingleDecodingResult::get);

  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const py::object& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt) {
        return async_decode(
            MediaType::Video,
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options, cuda_device_index},
            get_video_filter_description(
                get_frame_rate(frame_rate), width, height, pix_fmt));
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none());

  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc) {
        return async_decode(
            MediaType::Video,
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options, cuda_device_index},
            filter_desc);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string());

  m.def(
      "decode_image",
      [](const std::string& src,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt) {
        return async_decode_image(
            src,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            get_video_filter_description(std::nullopt, width, height, pix_fmt));
      },
      py::arg("src"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none());

  m.def(
      "decode_image",
      [](const std::string& src,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc) {
        return async_decode_image(
            src,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            filter_desc);
      },
      py::arg("src"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string());

  m.def(
      "batch_decode_image",
      [](const std::vector<std::string>& srcs,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt) {
        return async_batch_decode_image(
            srcs,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            get_video_filter_description(std::nullopt, width, height, pix_fmt));
      },
      py::arg("srcs"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none());

  m.def(
      "batch_decode_image",
      [](const std::vector<std::string>& srcs,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc) {
        return async_batch_decode_image(
            srcs,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            filter_desc);
      },
      py::arg("srcs"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string());

  m.def(
      "decode_audio",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& sample_rate,
         const std::optional<int>& num_channels,
         const std::optional<std::string>& sample_fmt) {
        return async_decode(
            MediaType::Audio,
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            get_audio_filter_description(
                sample_rate, num_channels, sample_fmt));
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("sample_rate") = py::none(),
      py::arg("num_channels") = py::none(),
      py::arg("sample_fmt") = py::none());

  m.def(
      "decode_audio",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::string& filter_desc) {
        return async_decode(
            MediaType::Audio,
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            filter_desc);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("filter_desc") = std::string());

  m.def(
      "decode_video_nvdec",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const int cuda_device_index,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt) {
        return async_decode_nvdec(
            src,
            timestamps,
            cuda_device_index,
            adoptor,
            {format, format_options, buffer_size},
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
            width,
            height,
            pix_fmt);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("cuda_device_index"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("crop_left") = 0,
      py::arg("crop_top") = 0,
      py::arg("crop_right") = 0,
      py::arg("crop_bottom") = 0,
      py::arg("width") = -1,
      py::arg("height") = -1,
      py::arg("pix_fmt") = "rgba");

  m.def(
      "decode_image_nvdec",
      [](const std::string& src,
         const int cuda_device_index,
         const std::shared_ptr<SourceAdoptor>& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt) {
        return async_decode_image_nvdec(
            src,
            cuda_device_index,
            adoptor,
            {format, format_options, buffer_size},
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
            width,
            height,
            pix_fmt);
      },
      py::arg("src"),
      py::arg("cuda_device_index"),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("crop_left") = 0,
      py::arg("crop_top") = 0,
      py::arg("crop_right") = 0,
      py::arg("crop_bottom") = 0,
      py::arg("width") = -1,
      py::arg("height") = -1,
      py::arg("pix_fmt") = "rgba");
}
} // namespace spdl::core
