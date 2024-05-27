#include <libspdl/core/conversion.h>
#include <libspdl/core/cuda.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
namespace {
template <MediaType media_type>
CPUBufferPtr convert(const FFmpegFramesPtr<media_type> frames) {
  nb::gil_scoped_release g;
  return convert_frames(frames.get());
}

template <MediaType media_type>
CUDABufferPtr convert_cuda(
    const FFmpegFramesPtr<media_type> frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator) {
  nb::gil_scoped_release g;
  return convert_to_cuda(
      convert_frames<media_type>(frames.get()),
      cuda_device_index,
      cuda_stream,
      cuda_allocator);
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
CPUBufferPtr batch_convert(std::vector<FFmpegFramesPtr<media_type>>&& frames) {
  nb::gil_scoped_release g;
  return convert_frames(_ref(frames));
}

template <MediaType media_type>
CUDABufferPtr batch_convert_cuda(
    std::vector<FFmpegFramesPtr<media_type>>&& frames,
    int cuda_device_index,
    const uintptr_t cuda_stream,
    const std::optional<cuda_allocator>& cuda_allocator) {
  nb::gil_scoped_release g;
  return convert_to_cuda(
      convert_frames(_ref(frames)),
      cuda_device_index,
      cuda_stream,
      cuda_allocator);
}
} // namespace

void register_conversion(nb::module_& m) {
  ////////////////////////////////////////////////////////////////////////////////
  // CPU conversion
  ////////////////////////////////////////////////////////////////////////////////
  m.def("convert_frames", &convert<MediaType::Audio>, nb::arg("frames"));
  m.def("convert_frames", &convert<MediaType::Video>, nb::arg("frames"));
  m.def("convert_frames", &convert<MediaType::Image>, nb::arg("frames"));

  m.def("convert_frames", &batch_convert<MediaType::Audio>, nb::arg("frames"));
  m.def("convert_frames", &batch_convert<MediaType::Video>, nb::arg("frames"));
  m.def("convert_frames", &batch_convert<MediaType::Image>, nb::arg("frames"));


  ////////////////////////////////////////////////////////////////////////////////
  // CUDA conversion
  ////////////////////////////////////////////////////////////////////////////////
  m.def(
      "convert_frames_cuda",
      &convert_cuda<MediaType::Audio>,
      nb::arg("frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  m.def(
      "convert_frames_cuda",
      &convert_cuda<MediaType::Video>,
      nb::arg("frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  m.def(
      "convert_frames_cuda",
      &convert_cuda<MediaType::Image>,
      nb::arg("frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  ////////////////////////////////////////////////////////////////////////////////
  // CUDA conversion (batch)
  ////////////////////////////////////////////////////////////////////////////////

  m.def(
      "convert_frames_cuda",
      &batch_convert_cuda<MediaType::Audio>,
      nb::arg("frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  m.def(
      "convert_frames_cuda",
      &batch_convert_cuda<MediaType::Video>,
      nb::arg("frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());

  m.def(
      "convert_frames_cuda",
      &batch_convert_cuda<MediaType::Image>,
      nb::arg("frames"),
#if NB_VERSION_MAJOR >= 2
      nb::kw_only(),
#endif
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none());
}
} // namespace spdl::core
