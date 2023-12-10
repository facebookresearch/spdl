#include <folly/logging/xlog.h>
#include <libspdl/processors.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace py = pybind11;

namespace spdl {
namespace {

std::optional<AVRational> to_rational(std::optional<std::tuple<int, int>> src) {
  if (!src) {
    return {};
  }
  AVRational r;
  std::tie(r.num, r.den) = *src;
  return {r};
}

py::buffer_info get_buffer(VideoBuffer& b) {
  if (b.channel_last) {
    return {
        b.data.data(),
        sizeof(uint8_t),
        py::format_descriptor<uint8_t>::format(),
        4,
        {b.n, b.h, b.w, b.c},
        {sizeof(uint8_t) * b.c * b.w * b.h,
         sizeof(uint8_t) * b.c * b.w,
         sizeof(uint8_t) * b.c,
         sizeof(uint8_t)}};
  }
  return {
      b.data.data(),
      sizeof(uint8_t),
      py::format_descriptor<uint8_t>::format(),
      4,
      {b.n, b.c, b.h, b.w},
      {sizeof(uint8_t) * b.w * b.h * b.c,
       sizeof(uint8_t) * b.w * b.h,
       sizeof(uint8_t) * b.w,
       sizeof(uint8_t)}};
}

PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  py::class_<VideoBuffer>(
      m, "VideoBuffer", py::buffer_protocol(), py::module_local())
      .def_buffer(get_buffer);
  py::class_<Engine>(m, "Engine", py::module_local())
      .def(
          py::init(
              [](size_t num_io_threads,
                 size_t num_decoding_threads,
                 size_t frame_queue_size,
                 std::optional<std::tuple<int, int>> frame_rate = std::nullopt,
                 std::optional<int> width = std::nullopt,
                 std::optional<int> height = std::nullopt,
                 std::optional<std::string> pix_fmt = std::nullopt) {
                return new Engine{
                    num_io_threads,
                    num_decoding_threads,
                    frame_queue_size,
                    to_rational(frame_rate),
                    width,
                    height,
                    pix_fmt};
              }),
          py::arg("num_io_threads"),
          py::arg("num_decoding_threads"),
          py::arg("frame_queue_size"),
          py::arg("frame_rate") = py::none(),
          py::arg("width") = py::none(),
          py::arg("height") = py::none(),
          py::arg("pix_fmt") = py::none())
      .def(
          "enqueue",
          [](Engine& self,
             const std::string& url,
             const std::vector<double>& timestamps) {
            self.enqueue({url, timestamps});
          })
      .def("dequeue", &Engine::dequeue);
}
} // namespace
} // namespace spdl
