#include <pybind11/pybind11.h>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace py = pybind11;

namespace spdl::core {
void register_pybind(py::module&);
} // namespace spdl::core

namespace {
PYBIND11_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  spdl::core::register_pybind(m);
}
} // namespace
