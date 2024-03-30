#include <libspdl/core/conversion.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
void register_conversion(py::module& m) {
  /////////////////////////////////////////////////////////////////////////////
  // Synchronous conversion
  /////////////////////////////////////////////////////////////////////////////
  m.def("convert_to_buffer", &convert_audio_frames);
  m.def(
      "convert_to_buffer",
      &convert_vision_frames<MediaType::Video, /*cpu_only=*/false>);
  m.def(
      "convert_to_buffer",
      &convert_vision_frames<MediaType::Image, /*cpu_only=*/false>);
  m.def("convert_to_buffer", &convert_batch_image_frames</*cpu_only=*/false>);
  m.def("convert_to_buffer", &convert_nvdec_frames<MediaType::Video>);
  m.def("convert_to_buffer", &convert_nvdec_frames<MediaType::Image>);
  m.def("convert_to_buffer", &convert_nvdec_batch_image_frames);

  m.def("convert_to_cpu_buffer", &convert_audio_frames);
  m.def(
      "convert_to_cpu_buffer",
      &convert_vision_frames<MediaType::Video, /*cpu_only=*/true>);
  m.def(
      "convert_to_cpu_buffer",
      &convert_vision_frames<MediaType::Image, /*cpu_only=*/true>);
  m.def(
      "convert_to_cpu_buffer", &convert_batch_image_frames</*cpu_only=*/true>);

  /////////////////////////////////////////////////////////////////////////////
  // Async conversion
  /////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_convert_audio_cpu",
      &async_convert_frames<MediaType::Audio, /*cpu_only=*/true>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_video_cpu",
      &async_convert_frames<MediaType::Video, /*cpu_only=*/true>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_image_cpu",
      &async_convert_frames<MediaType::Image, /*cpu_only=*/true>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_convert_audio",
      &async_convert_frames<MediaType::Audio, /*cpu_only=*/true>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_video",
      &async_convert_frames<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_image",
      &async_convert_frames<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_video_nvdec",
      &async_convert_nvdec_frames<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_image_nvdec",
      &async_convert_nvdec_frames<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image",
      &async_batch_convert_frames</*cpu_only=*/false>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image_cpu",
      &async_batch_convert_frames</*cpu_only=*/true>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image_nvdec",
      &async_batch_convert_nvdec_frames,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
}
} //  namespace spdl::core
