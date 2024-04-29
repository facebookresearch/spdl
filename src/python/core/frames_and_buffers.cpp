#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace nb = nanobind;

namespace spdl::core {
namespace {
////////////////////////////////////////////////////////////////////////////////
// Array interface supplements
////////////////////////////////////////////////////////////////////////////////
std::string get_typestr(const ElemClass elem_class, size_t depth) {
  const auto key = [&]() {
    switch (elem_class) {
      case ElemClass::UInt:
        return "u";
      case ElemClass::Int:
        return "i";
      case ElemClass::Float:
        return "f";
      default:
        throw std::runtime_error(
            fmt::format("Unsupported class {}.", int(elem_class)));
    }
  }();
  return fmt::format("|{}{}", key, depth);
}

nb::dict get_array_interface(Buffer* b) {
  auto typestr = get_typestr(b->elem_class, b->depth);
  nb::dict ret;
  ret["version"] = 3;
  ret["shape"] = nb::tuple(nb::cast(b->shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b->data(), false};
  ret["strides"] = nb::none();
  ret["descr"] =
      std::vector<std::tuple<std::string, std::string>>{{"", typestr}};
  return ret;
}

#ifdef SPDL_USE_CUDA
nb::dict get_cuda_array_interface(CUDABuffer* b) {
  auto typestr = get_typestr(b->elem_class, b->depth);
  nb::dict ret;
  ret["version"] = 2;
  ret["shape"] = nb::tuple(nb::cast(b->shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b->data(), false};
  ret["strides"] = nb::none();
  ret["stream"] = b->get_cuda_stream();
  return ret;
}
#endif

#ifdef SPDL_USE_NVCODEC
nb::dict get_cuda_array_interface(CUDABuffer2DPitch& b) {
  if (!b.p) {
    throw std::runtime_error("CUDA buffer is empty.");
  }
  auto typestr = get_typestr(ElemClass::UInt, 1);
  nb::dict ret;
  ret["version"] = 2;
  ret["shape"] = nb::tuple(nb::cast(b.get_shape()));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.p, false};
  auto hp = b.h * b.pitch;
  ret["strides"] = b.is_image
      ? nb::tuple(nb::cast(
            b.channel_last ? std::vector<size_t>{b.pitch, b.c * b.bpp, b.bpp}
                           : std::vector<size_t>{hp, b.pitch, b.bpp}))
      : nb::tuple(nb::cast(
            b.channel_last
                ? std::vector<size_t>{hp, b.pitch, b.c * b.bpp, b.bpp}
                : std::vector<size_t>{b.c * hp, hp, b.pitch, b.bpp}));
  ret["stream"] = nb::none();
  return ret;
}
#endif

} // namespace

void register_frames_and_buffers(nb::module_& m) {
  auto _Buffer = nb::class_<Buffer>(m, "Buffer");

  auto _CUDABuffer2DPitch =
      nb::class_<CUDABuffer2DPitch>(m, "CUDABuffer2DPitch");

  auto _FFmpegAudioFrames =
      nb::class_<FFmpegAudioFrames>(m, "FFmpegAudioFrames");

  auto _FFmpegVideoFrames =
      nb::class_<FFmpegVideoFrames>(m, "FFmpegVideoFrames");

  auto _FFmpegImageFrames =
      nb::class_<FFmpegImageFrames>(m, "FFmpegImageFrames");

  auto _NvDecVideoFrames = nb::class_<NvDecVideoFrames>(m, "NvDecVideoFrames");

  auto _NvDecImageFrames = nb::class_<NvDecImageFrames>(m, "NvDecImageFrames");

#ifdef SPDL_USE_CUDA
#define IF_CUDABUFFER_ENABLED(x) x
#else
#define IF_CUDABUFFER_ENABLED(x)                                         \
  [](const CUDABuffer&) {                                                \
    throw std::runtime_error("SPDL is not compiled with CUDA support."); \
  }
#endif

  _Buffer
      .def_prop_ro("is_cuda", [](const Buffer& self) { return self.is_cuda(); })
      .def_prop_ro("shape", [](const Buffer& self) { return self.shape; })
      .def_prop_ro(
          "__array_interface__",
          [](Buffer& self) {
            if (self.is_cuda()) {
              throw std::runtime_error(
                  "__array_interface__ is only available for CPU buffers.");
            }
            return get_array_interface(&self);
          })
      .def_prop_ro(
          "__cuda_array_interface__", IF_CUDABUFFER_ENABLED([](Buffer& self) {
            if (!self.is_cuda()) {
              throw std::runtime_error(
                  "__cuda_array_interface__ is only available for CUDA buffers.");
            }
            return get_cuda_array_interface(static_cast<CUDABuffer*>(&self));
          }))
      .def_prop_ro(
          "device_index", IF_CUDABUFFER_ENABLED([](const Buffer& self) {
            if (!self.is_cuda()) {
              throw std::runtime_error(
                  "__cuda_array_interface__ is only available for CUDA buffers.");
            }
            return (static_cast<const CUDABuffer*>(&self))->device_index;
          }));

#ifdef SPDL_USE_NVCODEC
#define IF_CUDABUFFER2_ENABLED(x) x
#else
#define IF_CUDABUFFER2_ENABLED(x)                                         \
  [](const CUDABuffer2DPitch&) {                                          \
    throw std::runtime_error("SPDL is not compiled with NVDEC support."); \
  }
#endif

  _CUDABuffer2DPitch
      .def_prop_ro(
          "channel_last",
          IF_CUDABUFFER2_ENABLED(
              [](const CUDABuffer2DPitch& self) { return self.channel_last; }))
      .def_prop_ro(
          "ndim", IF_CUDABUFFER2_ENABLED([](const CUDABuffer2DPitch& self) {
            return self.get_shape().size();
          }))
      .def_prop_ro(
          "shape", IF_CUDABUFFER2_ENABLED(&CUDABuffer2DPitch::get_shape))
      .def_prop_ro(
          "is_cuda", IF_CUDABUFFER2_ENABLED([](const CUDABuffer2DPitch& self) {
            return true;
          }))
      .def_prop_ro(
          "device_index",
          IF_CUDABUFFER2_ENABLED(
              [](const CUDABuffer2DPitch& self) { return self.device_index; }))
      .def_prop_ro(
          "__cuda_array_interface__",
          IF_CUDABUFFER2_ENABLED([](CUDABuffer2DPitch& self) {
            return get_cuda_array_interface(self);
          }));

  _FFmpegAudioFrames
      .def_prop_ro(
          "num_frames",
          [](FFmpegAudioFrames& self) { return self.get_num_frames(); })
      .def_prop_ro(
          "sample_rate",
          [](FFmpegAudioFrames& self) { return self.get_sample_rate(); })
      .def_prop_ro(
          "num_channels",
          [](FFmpegAudioFrames& self) { return self.get_num_channels(); })
      .def_prop_ro(
          "format",
          [](FFmpegAudioFrames& self) { return self.get_media_format_name(); })
      .def(
          "__len__",
          [](FFmpegAudioFrames& self) { return self.get_num_frames(); })
      .def(
          "__repr__",
          [](const FFmpegAudioFrames& self) {
            return fmt::format(
                "FFmpegAudioFrames<num_frames={}, sample_format=\"{}\", sample_rate={}, num_channels={}, pts={}>",
                self.get_num_frames(),
                self.get_media_format_name(),
                self.get_sample_rate(),
                self.get_num_channels(),
                double(self.get_frames().front()->pts) * self.time_base.num /
                    self.time_base.den);
          })
      .def("clone", [](const FFmpegAudioFrames& self) { return clone(self); });

  _FFmpegVideoFrames
      .def_prop_ro(
          "num_frames",
          [](FFmpegVideoFrames& self) { return self.get_num_frames(); })
      .def_prop_ro(
          "num_planes",
          [](FFmpegVideoFrames& self) { return self.get_num_planes(); })
      .def_prop_ro(
          "width", [](FFmpegVideoFrames& self) { return self.get_width(); })
      .def_prop_ro(
          "height", [](FFmpegVideoFrames& self) { return self.get_height(); })
      .def_prop_ro(
          "format",
          [](FFmpegVideoFrames& self) { return self.get_media_format_name(); })
      .def(
          "__len__",
          [](FFmpegVideoFrames& self) { return self.get_num_frames(); })
      .def(
          "__getitem__",
          [](const FFmpegVideoFrames& self, const nb::slice& slice) {
            auto [start, stop, step, len] =
                slice.compute(self.get_num_frames());
            return self.slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step));
          })
      .def(
          "__getitem__",
          [](const FFmpegVideoFrames& self, int64_t i) {
            return self.slice(i);
          })
      .def(
          "__getitem__",
          [](const FFmpegVideoFrames& self, std::vector<int64_t> idx) {
            return self.slice(idx);
          })
      .def(
          "_get_pts",
          [](const FFmpegVideoFrames& self) -> std::vector<double> {
            std::vector<double> ret;
            auto base = self.time_base;
            for (auto& frame : self.get_frames()) {
              ret.push_back(double(frame->pts) * base.num / base.den);
            }
            return ret;
          })
      .def(
          "__repr__",
          [](const FFmpegVideoFrames& self) {
            return fmt::format(
                "FFmpegVideoFrames<num_frames={}, pixel_format=\"{}\", num_planes={}, width={}, height={}, pts={}>",
                self.get_num_frames(),
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height(),
                double(self.get_frames().front()->pts) * self.time_base.num /
                    self.time_base.den);
          })
      .def("clone", [](const FFmpegVideoFrames& self) { return clone(self); });

  _FFmpegImageFrames
      .def_prop_ro(
          "num_planes",
          [](const FFmpegImageFrames& self) { return self.get_num_planes(); })
      .def_prop_ro(
          "width",
          [](const FFmpegImageFrames& self) { return self.get_width(); })
      .def_prop_ro(
          "height",
          [](const FFmpegImageFrames& self) { return self.get_height(); })
      .def_prop_ro(
          "format",
          [](const FFmpegImageFrames& self) {
            return self.get_media_format_name();
          })
      .def(
          "__repr__",
          [](const FFmpegImageFrames& self) {
            return fmt::format(
                "FFmpegImageFrames<pixel_format=\"{}\", num_planes={}, width={}, height={}>",
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height());
          })
      .def("clone", [](const FFmpegImageFrames& self) { return clone(self); })
      .def_prop_ro("pts", [](const FFmpegImageFrames& self) -> double {
        auto base = self.time_base;
        auto& frame = self.get_frames().at(0);
        return double(frame->pts) * base.num / base.den;
      });

#ifdef SPDL_USE_NVCODEC
#define IF_NVDECVIDEOFRAMES_ENABLED(x) x
#else
#define IF_NVDECVIDEOFRAMES_ENABLED(x)                                    \
  [](const NvDecVideoFrames&) {                                           \
    throw std::runtime_error("SPDL is not compiled with NVDEC support."); \
  }
#endif

  // TODO: Add __repr__
  _NvDecVideoFrames
      .def_prop_ro(
          "channel_last",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->channel_last;
          }))
      .def_prop_ro(
          "ndim", IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->get_shape().size();
          }))
      .def_prop_ro(
          "shape",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->get_shape();
          }))
      .def_prop_ro(
          "format",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.get_media_format_name();
          }))
      .def_prop_ro("is_cuda", [](const NvDecVideoFrames& self) { return true; })
      .def_prop_ro(
          "__cuda_array_interface__",
          IF_NVDECVIDEOFRAMES_ENABLED([](NvDecVideoFrames& self) {
            return get_cuda_array_interface(*self.buffer);
          }))
      .def(
          "__len__",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->get_shape()[0];
          }))
      .def(
          "clone",
          IF_NVDECVIDEOFRAMES_ENABLED(
              ([](const NvDecVideoFrames& self) { return clone(self); })))
      .def_prop_ro(
          "device_index",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFrames& self) {
            return self.buffer->device_index;
          }));

  // TODO: Add __repr__
  _NvDecImageFrames
      .def_prop_ro(
          "channel_last",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->channel_last;
          }))
      .def_prop_ro(
          "ndim", IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->get_shape().size();
          }))
      .def_prop_ro(
          "shape",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->get_shape();
          }))
      .def_prop_ro(
          "format",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.get_media_format_name();
          }))
      .def_prop_ro("is_cuda", [](const NvDecImageFrames& self) { return true; })
      .def_prop_ro(
          "__cuda_array_interface__",
          IF_NVDECVIDEOFRAMES_ENABLED([](NvDecImageFrames& self) {
            return get_cuda_array_interface(*self.buffer);
          }))
      .def(
          "__len__",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->get_shape()[0];
          }))
      .def(
          "clone",
          IF_NVDECVIDEOFRAMES_ENABLED(
              ([](const NvDecImageFrames& self) { return clone(self); })))
      .def_prop_ro(
          "device_index",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFrames& self) {
            return self.buffer->device_index;
          }));
}
} // namespace spdl::core
