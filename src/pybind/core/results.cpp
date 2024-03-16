#include <libspdl/core/decoding.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spdl::core {

void register_results(py::module& m) {
  auto _DecodeImageResult = py::class_<DecodeResult<MediaType::Image>>(
      m, "DecodeImageResult", py::module_local());

  auto _DecodeImageNvDecResult =
      py::class_<DecodeNvDecResult<MediaType::Image>>(
          m, "DecodeImageNvDecResult", py::module_local());

  auto _BatchDecodeAudioResult =
      py::class_<BatchDecodeResult<MediaType::Audio>>(
          m, "BatchDecodeAudioResult", py::module_local());

  auto _BatchDecodeVideoResult =
      py::class_<BatchDecodeResult<MediaType::Video>>(
          m, "BatchDecodeVideoResult", py::module_local());

  auto _BatchDecodeImageResult =
      py::class_<BatchDecodeResult<MediaType::Image>>(
          m, "BatchDecodeImageResult", py::module_local());

  auto _BatchDecodeVideoNvDecResult =
      py::class_<BatchDecodeNvDecResult<MediaType::Video>>(
          m, "BatchDecodeVideoNvDecResult", py::module_local());

  auto _BatchDecodeImageNvDecResult =
      py::class_<BatchDecodeNvDecResult<MediaType::Image>>(
          m, "BatchDecodeImageNvDecResult", py::module_local());

  _DecodeImageResult.def("get", &DecodeResult<MediaType::Image>::get);

  _DecodeImageNvDecResult.def("get", &DecodeNvDecResult<MediaType::Image>::get);

  _BatchDecodeAudioResult.def(
      "get",
      &BatchDecodeResult<MediaType::Audio>::get,
      py::arg("strict") = true);

  _BatchDecodeVideoResult.def(
      "get",
      &BatchDecodeResult<MediaType::Video>::get,
      py::arg("strict") = true);

  _BatchDecodeImageResult.def(
      "get",
      &BatchDecodeResult<MediaType::Image>::get,
      py::arg("strict") = true);

  _BatchDecodeVideoNvDecResult.def(
      "get",
      &BatchDecodeNvDecResult<MediaType::Video>::get,
      py::arg("strict") = true);

  _BatchDecodeImageNvDecResult.def(
      "get",
      &BatchDecodeNvDecResult<MediaType::Image>::get,
      py::arg("strict") = true);
}
} // namespace spdl::core
