#include <folly/logging/xlog.h>
#include <libspdl/processors.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

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

namespace py = pybind11;

PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
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
