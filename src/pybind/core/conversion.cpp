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
  m.def("convert_to_buffer", &convert_visual_frames<MediaType::Video>);
  m.def("convert_to_buffer", &convert_visual_frames<MediaType::Image>);
  m.def("convert_to_buffer", &convert_batch_image_frames);
  m.def("convert_to_buffer", &convert_nvdec_frames<MediaType::Video>);
  m.def("convert_to_buffer", &convert_nvdec_frames<MediaType::Image>);
  m.def("convert_to_buffer", &convert_nvdec_batch_image_frames);

  m.def("convert_to_cpu_buffer", &convert_audio_frames);
  m.def(
      "convert_to_cpu_buffer",
      &convert_visual_frames_to_cpu_buffer<MediaType::Video>);
  m.def(
      "convert_to_cpu_buffer",
      &convert_visual_frames_to_cpu_buffer<MediaType::Image>);
  m.def("convert_to_cpu_buffer", &convert_batch_image_frames_to_cpu_buffer);

  /////////////////////////////////////////////////////////////////////////////
  // Async conversion
  /////////////////////////////////////////////////////////////////////////////
  m.def(
      "async_convert_cpu",
      &async_convert_frames_to_cpu<MediaType::Audio>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_cpu",
      &async_convert_frames_to_cpu<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_cpu",
      &async_convert_frames_to_cpu<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_convert",
      &async_convert_frames<MediaType::Audio>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert",
      &async_convert_frames<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert",
      &async_convert_frames<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert",
      &async_convert_nvdec_frames<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert",
      &async_convert_nvdec_frames<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert",
      &async_batch_convert_frames,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert",
      &async_batch_convert_nvdec_frames,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("index") = py::none(),
      py::arg("executor") = nullptr);
}
} //  namespace spdl::core
