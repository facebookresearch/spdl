#include <pybind11/pybind11.h>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace py = pybind11;

namespace spdl::core {
void register_adoptors(py::module&);
void register_tracing(py::module&);
void register_utils(py::module&);
void register_frames_and_buffers(py::module&);
void register_future(py::module&);
void register_pybind(py::module&);
} // namespace spdl::core

namespace {
PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  spdl::core::register_adoptors(m);
  spdl::core::register_tracing(m);
  spdl::core::register_utils(m);
  spdl::core::register_frames_and_buffers(m);
  spdl::core::register_future(m);
  spdl::core::register_pybind(m);
}
} // namespace
