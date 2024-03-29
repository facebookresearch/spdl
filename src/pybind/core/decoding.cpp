#include <libspdl/core/decoding.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/result.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
std::optional<Rational> get_frame_rate(const py::object& frame_rate) {
  if (frame_rate.is(py::none())) {
    return std::nullopt;
  }
  py::object Fraction = py::module_::import("fractions").attr("Fraction");
  py::object r = Fraction(frame_rate);
  return {Rational{
      r.attr("numerator").cast<int>(), r.attr("denominator").cast<int>()}};
}
} // namespace

void register_decoding(py::module& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Async decoding - FFMPEG
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_sleep",
      &async_sleep,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("duration"),
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_audio",
      [](std::function<void(FFmpegAudioFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         AudioPacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::string& filter_desc,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        return async_decode<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options},
            filter_desc,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("filter_desc") = std::string(),
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_audio",
      [](std::function<void(FFmpegAudioFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         AudioPacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& sample_rate,
         const std::optional<int>& num_channels,
         const std::optional<std::string>& sample_fmt,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        return async_decode<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options},
            get_audio_filter_description(sample_rate, num_channels, sample_fmt),
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("sample_rate") = py::none(),
      py::arg("num_channels") = py::none(),
      py::arg("sample_fmt") = py::none(),
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_video",
      [](std::function<void(FFmpegVideoFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         VideoPacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        return async_decode<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options, cuda_device_index},
            filter_desc,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string(),
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_video",
      [](std::function<void(FFmpegVideoFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         VideoPacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::optional<Rational>& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        return async_decode<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options, cuda_device_index},
            get_video_filter_description(frame_rate, width, height, pix_fmt),
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none(),
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_image",
      [](std::function<void(FFmpegImageFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         ImagePacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        return async_decode<MediaType::Image>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options, cuda_device_index},
            filter_desc,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string(),
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_image",
      [](std::function<void(FFmpegImageFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         ImagePacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::optional<Rational>& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        return async_decode<MediaType::Image>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options, cuda_device_index},
            get_video_filter_description(frame_rate, width, height, pix_fmt),
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none(),
      py::arg("decode_executor") = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_decode_video_nvdec",
      [](std::function<void(NvDecVideoFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         VideoPacketsWrapperPtr packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr decode_executor) {
        return async_decode_nvdec<MediaType::Video>(
            set_result,
            notify_exception,
            packets,
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("cuda_device_index"),
      py::arg("crop_left") = 0,
      py::arg("crop_top") = 0,
      py::arg("crop_right") = 0,
      py::arg("crop_bottom") = 0,
      py::arg("width") = -1,
      py::arg("height") = -1,
      py::arg("pix_fmt") = "rgba",
      py::arg("decode_executor") = nullptr);

  m.def(
      "async_decode_image_nvdec",
      [](std::function<void(NvDecImageFramesWrapperPtr)> set_result,
         std::function<void()> notify_exception,
         ImagePacketsWrapperPtr packets,
         const int cuda_device_index,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr decode_executor) {
        return async_decode_nvdec<MediaType::Image>(
            set_result,
            notify_exception,
            packets,
            cuda_device_index,
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("cuda_device_index"),
      py::arg("crop_left") = 0,
      py::arg("crop_top") = 0,
      py::arg("crop_right") = 0,
      py::arg("crop_bottom") = 0,
      py::arg("width") = -1,
      py::arg("height") = -1,
      py::arg("pix_fmt") = "rgba",
      py::arg("decode_executor") = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Synchronous decoding
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const py::object& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode<MediaType::Video>(
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options, cuda_device_index},
            get_video_filter_description(
                get_frame_rate(frame_rate), width, height, pix_fmt),
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
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
      py::arg("pix_fmt") = py::none(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_video",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode<MediaType::Video>(
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options, cuda_device_index},
            filter_desc,
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_image",
      [](const std::string& src,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode_image(
            src,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            get_video_filter_description(std::nullopt, width, height, pix_fmt),
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_image",
      [](const std::string& src,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode_image(
            src,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            filter_desc,
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "batch_decode_image",
      [](const std::vector<std::string>& srcs,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::batch_decode_image(
            srcs,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            get_video_filter_description(std::nullopt, width, height, pix_fmt),
            demux_executor,
            decode_executor);
      },
      py::arg("srcs"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "batch_decode_image",
      [](const std::vector<std::string>& srcs,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const int cuda_device_index,
         const std::string& filter_desc,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::batch_decode_image(
            srcs,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            filter_desc,
            demux_executor,
            decode_executor);
      },
      py::arg("srcs"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("cuda_device_index") = -1,
      py::arg("filter_desc") = std::string(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_audio",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& sample_rate,
         const std::optional<int>& num_channels,
         const std::optional<std::string>& sample_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode<MediaType::Audio>(
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            get_audio_filter_description(sample_rate, num_channels, sample_fmt),
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("sample_rate") = py::none(),
      py::arg("num_channels") = py::none(),
      py::arg("sample_fmt") = py::none(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_audio",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::string& filter_desc,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode<MediaType::Audio>(
            src,
            timestamps,
            adoptor,
            {format, format_options, buffer_size},
            {decoder, decoder_options},
            filter_desc,
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::kw_only(),
      py::arg("adoptor") = nullptr,
      py::arg("format") = py::none(),
      py::arg("format_options") = py::none(),
      py::arg("buffer_size") = SPDL_DEFAULT_BUFFER_SIZE,
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("filter_desc") = std::string(),
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_video_nvdec",
      [](const std::string& src,
         const std::vector<std::tuple<double, double>>& timestamps,
         const int cuda_device_index,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode_video_nvdec(
            src,
            timestamps,
            cuda_device_index,
            adoptor,
            {format, format_options, buffer_size},
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("timestamps"),
      py::arg("cuda_device_index"),
      py::kw_only(),
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
      py::arg("pix_fmt") = "rgba",
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "decode_image_nvdec",
      [](const std::string& src,
         const int cuda_device_index,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::decode_image_nvdec(
            src,
            cuda_device_index,
            adoptor,
            {format, format_options, buffer_size},
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("cuda_device_index"),
      py::kw_only(),
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
      py::arg("pix_fmt") = "rgba",
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);

  m.def(
      "batch_decode_image_nvdec",
      [](const std::vector<std::string>& srcs,
         const int cuda_device_index,
         const SourceAdoptorPtr& adoptor,
         const std::optional<std::string>& format,
         const std::optional<OptionDict>& format_options,
         int buffer_size,
         int crop_left,
         int crop_top,
         int crop_right,
         int crop_bottom,
         int width,
         int height,
         const std::optional<std::string>& pix_fmt,
         ThreadPoolExecutorPtr demux_executor,
         ThreadPoolExecutorPtr decode_executor) {
        return decoding::batch_decode_image_nvdec(
            srcs,
            cuda_device_index,
            adoptor,
            {format, format_options, buffer_size},
            {static_cast<short>(crop_left),
             static_cast<short>(crop_top),
             static_cast<short>(crop_right),
             static_cast<short>(crop_bottom)},
            width,
            height,
            pix_fmt,
            demux_executor,
            decode_executor);
      },
      py::arg("src"),
      py::arg("cuda_device_index"),
      py::kw_only(),
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
      py::arg("pix_fmt") = "rgba",
      py::arg("demux_executor") = nullptr,
      py::arg("decode_executor") = nullptr);
}
} // namespace spdl::core
