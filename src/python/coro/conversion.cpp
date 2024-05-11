#include <libspdl/coro/conversion.h>

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

namespace spdl::coro {
void register_conversion(nb::module_& m) {
  m.def(
      "async_convert_audio",
      &async_convert_frames<MediaType::Audio>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_audio_cuda",
      &async_convert_frames_cuda<MediaType::Audio>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_video",
      &async_convert_frames<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_video_cuda",
      &async_convert_frames_cuda<MediaType::Video>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_image",
      &async_convert_frames<MediaType::Image>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_image_cuda",
      &async_convert_frames_cuda<MediaType::Image>,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image",
      &async_batch_convert_frames,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("executor") = nullptr);
  m.def(
      "async_convert_batch_image_cuda",
      &async_batch_convert_frames_cuda,
      nb::arg("set_result"),
      nb::arg("notify_exception"),
      nb::arg("frames"),
      // nb::kw_only(),
      nb::arg("cuda_device_index"),
      nb::arg("cuda_stream") = 0,
      nb::arg("cuda_allocator") = nb::none(),
      nb::arg("executor") = nullptr);
}
} //  namespace spdl::coro
