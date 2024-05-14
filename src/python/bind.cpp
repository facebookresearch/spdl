#include <nanobind/nanobind.h>

#ifndef SPDL_FFMPEG_EXT_NAME
#error SPDL_FFMPEG_EXT_NAME must be defined.
#endif

namespace nb = nanobind;

namespace spdl::core {
void register_types(nb::module_&);
void register_adaptors(nb::module_&);
void register_packets(nb::module_&);
void register_frames(nb::module_&);
void register_buffers(nb::module_&);
void register_tracing(nb::module_&);
void register_utils(nb::module_&);
} // namespace spdl::core

namespace spdl::coro {
void register_future(nb::module_&);
void register_executor(nb::module_&);
void register_demuxing(nb::module_&);
void register_decoding(nb::module_&);
void register_conversion(nb::module_& m);
void register_encoding(nb::module_&);
} // namespace spdl::coro

namespace {
NB_MODULE(SPDL_FFMPEG_EXT_NAME, m) {
  spdl::core::register_types(m);
  spdl::core::register_adaptors(m);
  spdl::core::register_packets(m);
  spdl::core::register_frames(m);
  spdl::core::register_buffers(m);
  spdl::core::register_tracing(m);
  spdl::core::register_utils(m);
  // coro
  spdl::coro::register_future(m);
  spdl::coro::register_executor(m);
  spdl::coro::register_conversion(m);
  spdl::coro::register_demuxing(m);
  spdl::coro::register_decoding(m);
  spdl::coro::register_encoding(m);
}
} // namespace
