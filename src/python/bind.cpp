#include <nanobind/nanobind.h>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace nb = nanobind;

namespace spdl::core {
void register_adaptors(nb::module_&);
void register_conversion(nb::module_& m);
void register_tracing(nb::module_&);
void register_utils(nb::module_&);
void register_executor(nb::module_&);
void register_frames_and_buffers(nb::module_&);
void register_future(nb::module_&);
void register_packets(nb::module_&);
void register_demuxing(nb::module_&);
void register_decoding(nb::module_&);
} // namespace spdl::core

namespace {
NB_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  spdl::core::register_adaptors(m);
  spdl::core::register_conversion(m);
  spdl::core::register_tracing(m);
  spdl::core::register_utils(m);
  spdl::core::register_executor(m);
  spdl::core::register_frames_and_buffers(m);
  spdl::core::register_future(m);
  spdl::core::register_packets(m);
  spdl::core::register_demuxing(m);
  spdl::core::register_decoding(m);
}
} // namespace
