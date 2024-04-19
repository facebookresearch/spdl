#include <libspdl/core/decoding.h>
#include <libspdl/core/executor.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>

extern "C" {
#include <libavfilter/version.h>
}

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

// -----------------------------------------------------------------------------
std::string get_audio_filter_description(
    const std::optional<int>& sample_rate,
    const std::optional<int>& num_channels,
    const std::optional<std::string>& sample_fmt,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<int>& num_frames = std::nullopt,
    const std::optional<std::string>& filter_desc = std::nullopt) {
  std::vector<std::string> parts;
  if (num_channels || sample_fmt) {
    std::vector<std::string> aformat;
    if (num_channels) {
      aformat.emplace_back(
          fmt::format("channel_layouts={}c", num_channels.value()));
    }
    if (sample_fmt) {
      aformat.emplace_back(fmt::format("sample_fmts={}", sample_fmt.value()));
    }
    parts.push_back(fmt::format("aformat={}", fmt::join(aformat, ":")));
  }
  if (sample_rate) {
    parts.emplace_back(fmt::format("aresample={}", sample_rate.value()));
  }
  if (timestamp) {
    auto& ts = timestamp.value();
    std::vector<std::string> atrim;
    auto start = std::get<0>(ts), end = std::get<1>(ts);
    atrim.emplace_back(fmt::format("start={}", start));
    if (!std::isinf(end)) {
      atrim.emplace_back(fmt::format("end={}", end));
    }
    parts.push_back(fmt::format("atrim={}", fmt::join(atrim, ":")));
  }
  if (num_frames) {
    // Add endless silence then drop samples after the given frame
    parts.push_back("apad");
    parts.push_back(fmt::format("atrim=end_sample={}", num_frames.value()));
  }
  if (filter_desc) {
    parts.push_back(filter_desc.value());
  }
  return fmt::to_string(fmt::join(parts, ","));
}

// -----------------------------------------------------------------------------

std::string get_video_filter_description(
    const std::optional<Rational>& frame_rate,
    const std::optional<int>& width,
    const std::optional<int>& height,
    const std::optional<std::string>& pix_fmt,
    const std::optional<std::tuple<double, double>>& timestamp,
    const std::optional<int>& num_frames = std::nullopt,
    const std::optional<std::string>& pad_mode = std::nullopt,
    const std::optional<std::string>& filter_desc = std::nullopt) {
  std::vector<std::string> parts;
  if (frame_rate) {
    auto fr = frame_rate.value();
    parts.emplace_back(fmt::format("fps={}/{}", fr.num, fr.den));
  }
  if (timestamp) {
    auto& ts = timestamp.value();
    std::vector<std::string> atrim;
    auto start = std::get<0>(ts), end = std::get<1>(ts);
    atrim.emplace_back(fmt::format("start={}", start));
    if (!std::isinf(end)) {
      atrim.emplace_back(fmt::format("end={}", end));
    }
    parts.push_back(fmt::format("trim={}", fmt::join(atrim, ":")));
  }
  if (num_frames) {
    if (LIBAVFILTER_VERSION_INT < AV_VERSION_INT(7, 57, 100)) {
      throw std::runtime_error(
          "`num_frames` requires FFmpeg >= 4.2. "
          "Please upgrade your FFmpeg.");
    }
    // Add endless frame
    auto pad = [&]() -> std::string {
      if (!pad_mode) {
        return "tpad=stop=-1:stop_mode=clone";
      }
      return fmt::format(
          "tpad=stop=-1:stop_mode=add:color={}", pad_mode.value());
    }();
    parts.push_back(pad);
    // then drop frames after the given frame
    parts.push_back(fmt::format("trim=end_frame={}", num_frames.value()));
  }
  if (width || height) {
    std::vector<std::string> scale;
    if (width) {
      scale.emplace_back(fmt::format("width={}", width.value()));
    }
    if (height > 0) {
      scale.emplace_back(fmt::format("height={}", height.value()));
    }
    parts.push_back(fmt::format("scale={}", fmt::join(scale, ":")));
  }
  if (pix_fmt) {
    parts.push_back(fmt::format("format=pix_fmts={}", pix_fmt.value()));
  }

  if (filter_desc) {
    parts.push_back(filter_desc.value());
  }
  return fmt::to_string(fmt::join(parts, ","));
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
      py::arg("executor") = nullptr);

  m.def(
      "async_sleep_multi",
      &async_sleep_multi,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("duration"),
      py::arg("count"),
      py::arg("executor") = nullptr);

  m.def(
      "async_decode_audio",
      [](std::function<void(FFmpegAudioFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         AudioPacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<int>& sample_rate,
         const std::optional<int>& num_channels,
         const std::optional<std::string>& sample_fmt,
         const std::optional<int>& num_frames,
         const std::optional<std::string>& filter_desc,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        auto filter = get_audio_filter_description(
            sample_rate,
            num_channels,
            sample_fmt,
            packets->get_packets()->timestamp,
            num_frames,
            filter_desc);
        return async_decode<MediaType::Audio>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options},
            std::move(filter),
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
      py::arg("num_frames") = py::none(),
      py::arg("filter_desc") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_decode_video",
      [](std::function<void(FFmpegVideoFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         VideoPacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<Rational>& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         const std::optional<int>& num_frames,
         const std::optional<std::string>& pad_mode,
         const std::optional<std::string>& filter_desc,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        auto filter = get_video_filter_description(
            frame_rate,
            width,
            height,
            pix_fmt,
            packets->get_packets()->timestamp,
            num_frames,
            pad_mode,
            filter_desc);
        return async_decode<MediaType::Video>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options},
            std::move(filter),
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none(),
      py::arg("num_frames") = py::none(),
      py::arg("pad_mode") = py::none(),
      py::arg("filter_desc") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_decode_image",
      [](std::function<void(FFmpegImageFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
         ImagePacketsWrapperPtr packets,
         const std::optional<std::string>& decoder,
         const std::optional<OptionDict>& decoder_options,
         const std::optional<Rational>& frame_rate,
         const std::optional<int>& width,
         const std::optional<int>& height,
         const std::optional<std::string>& pix_fmt,
         const std::optional<std::string>& filter_desc,
         std::shared_ptr<ThreadPoolExecutor> decode_executor) {
        auto filter = get_video_filter_description(
            frame_rate,
            width,
            height,
            pix_fmt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            filter_desc);
        return async_decode<MediaType::Image>(
            std::move(set_result),
            std::move(notify_exception),
            packets,
            {decoder, decoder_options},
            std::move(filter),
            decode_executor);
      },
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("packets"),
      py::kw_only(),
      py::arg("decoder") = py::none(),
      py::arg("decoder_options") = py::none(),
      py::arg("frame_rate") = py::none(),
      py::arg("width") = py::none(),
      py::arg("height") = py::none(),
      py::arg("pix_fmt") = py::none(),
      py::arg("filter_desc") = py::none(),
      py::arg("executor") = nullptr);

  ////////////////////////////////////////////////////////////////////////////////
  // Asynchronous decoding - NVDEC
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_decode_video_nvdec",
      [](std::function<void(NvDecVideoFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
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
      py::arg("executor") = nullptr);

  m.def(
      "async_decode_image_nvdec",
      [](std::function<void(NvDecImageFramesWrapperPtr)> set_result,
         std::function<void(std::string, bool)> notify_exception,
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
      py::arg("executor") = nullptr);
}
} // namespace spdl::core
