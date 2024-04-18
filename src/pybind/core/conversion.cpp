#include <libspdl/core/conversion.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {
void register_conversion(py::module& m) {
  m.def(
      "async_convert_audio",
      &async_convert_frames<MediaType::Audio>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("cuda_device_index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_video",
      &async_convert_frames<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("cuda_device_index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_image",
      &async_convert_frames<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("cuda_device_index") = py::none(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image",
      &async_batch_convert_frames,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("cuda_device_index") = py::none(),
      py::arg("executor") = nullptr);

  m.def(
      "async_convert_video_nvdec",
      &async_convert_nvdec_frames<MediaType::Video>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_image_nvdec",
      &async_convert_nvdec_frames<MediaType::Image>,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image_nvdec",
      &async_batch_convert_nvdec_frames,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("frames"),
      py::kw_only(),
      py::arg("executor") = nullptr);

  m.def(
      "async_convert_to_cuda",
      &async_convert_to_cuda,
      py::arg("set_result"),
      py::arg("notify_exception"),
      py::arg("buffer"),
      py::arg("cuda_device_index"),
      py::kw_only(),
      py::arg("executor") = nullptr);
}
} //  namespace spdl::core
