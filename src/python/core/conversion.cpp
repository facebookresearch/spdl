#include <libspdl/core/conversion.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
CPUBufferPtr convert(
    const FFmpegFramesPtr<media_type> frames,
    bool pin_memory) {
  nb::gil_scoped_release g;
  return convert_frames(frames.get(), pin_memory);
}

template <MediaType media_type>
std::vector<const spdl::core::FFmpegFrames<media_type>*> _ref(
    std::vector<FFmpegFramesPtr<media_type>>& frames) {
  std::vector<const spdl::core::FFmpegFrames<media_type>*> ret;
  for (auto& frame : frames) {
    ret.push_back(frame.get());
  }
  return ret;
}

template <MediaType media_type>
CPUBufferPtr batch_convert(
    std::vector<FFmpegFramesPtr<media_type>>&& frames,
    bool pin_memory) {
  nb::gil_scoped_release g;
  return convert_frames(_ref(frames), pin_memory);
}
} // namespace

void register_conversion(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // Frame conversion
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "convert_frames",
      &convert<MediaType::Audio>,
      nb::arg("frames"),
      nb::arg("pin_memory") = false);
  m.def(
      "convert_frames",
      &convert<MediaType::Video>,
      nb::arg("frames"),
      nb::arg("pin_memory") = false);
  m.def(
      "convert_frames",
      &convert<MediaType::Image>,
      nb::arg("frames"),
      nb::arg("pin_memory") = false);

  m.def(
      "convert_frames",
      &batch_convert<MediaType::Audio>,
      nb::arg("frames"),
      nb::arg("pin_memory") = false);
  m.def(
      "convert_frames",
      &batch_convert<MediaType::Video>,
      nb::arg("frames"),
      nb::arg("pin_memory") = false);
  m.def(
      "convert_frames",
      &batch_convert<MediaType::Image>,
      nb::arg("frames"),
      nb::arg("pin_memory") = false);
}
} // namespace spdl::core
