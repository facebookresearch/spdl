#include <libspdl/core/buffer.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>

#include <fmt/core.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace py = pybind11;

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

py::dict get_array_interface(Buffer& b) {
  auto typestr = get_typestr(b.elem_class, b.depth);
  py::dict ret;
  ret["version"] = 3;
  ret["shape"] = py::tuple(py::cast(b.shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = py::none();
  ret["descr"] =
      std::vector<std::tuple<std::string, std::string>>{{"", typestr}};
  return ret;
}

#ifdef SPDL_USE_CUDA
py::dict get_cuda_array_interface(CUDABuffer& b) {
  auto typestr = get_typestr(b.elem_class, b.depth);
  py::dict ret;
  ret["version"] = 2;
  ret["shape"] = py::tuple(py::cast(b.shape));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.data(), false};
  ret["strides"] = py::none();
  ret["stream"] = b.get_cuda_stream();
  return ret;
}
#endif

#ifdef SPDL_USE_NVDEC
py::dict get_cuda_array_interface(CUDABuffer2DPitch& b) {
  if (!b.p) {
    throw std::runtime_error("CUDA buffer is empty.");
  }
  auto typestr = get_typestr(ElemClass::UInt, 1);
  py::dict ret;
  ret["version"] = 2;
  ret["shape"] = py::tuple(py::cast(b.get_shape()));
  ret["typestr"] = typestr;
  ret["data"] = std::tuple<size_t, bool>{(uintptr_t)b.p, false};
  auto hp = b.h * b.pitch;
  ret["strides"] = b.is_image
      ? py::tuple(py::cast(
            b.channel_last ? std::vector<size_t>{b.pitch, b.c * b.bpp, b.bpp}
                           : std::vector<size_t>{hp, b.pitch, b.bpp}))
      : py::tuple(py::cast(
            b.channel_last
                ? std::vector<size_t>{hp, b.pitch, b.c * b.bpp, b.bpp}
                : std::vector<size_t>{b.c * hp, hp, b.pitch, b.bpp}));
  ret["stream"] = py::none();
  return ret;
}
#endif

} // namespace

template <MediaType media_type>
using FFmpegFramesWrapper = FramesWrapper<media_type, FFmpegFramesPtr>;

using FFmpegAudioFramesWrapper = FFmpegFramesWrapper<MediaType::Audio>;
using FFmpegVideoFramesWrapper = FFmpegFramesWrapper<MediaType::Video>;
using FFmpegImageFramesWrapper = FFmpegFramesWrapper<MediaType::Image>;

template <MediaType media_type>
using NvDecFramesWrapper = FramesWrapper<media_type, NvDecFramesPtr>;

using NvDecAudioFramesWrapper = NvDecFramesWrapper<MediaType::Audio>;
using NvDecVideoFramesWrapper = NvDecFramesWrapper<MediaType::Video>;
using NvDecImageFramesWrapper = NvDecFramesWrapper<MediaType::Image>;

void register_frames_and_buffers(py::module& m) {
  auto _Buffer = py::class_<Buffer, BufferPtr>(m, "Buffer", py::module_local());

  auto _CPUBuffer = py::class_<CPUBuffer>(m, "CPUBuffer", py::module_local());

  auto _CUDABuffer =
      py::class_<CUDABuffer>(m, "CUDABuffer", py::module_local());

  auto _CUDABuffer2DPitch =
      py::class_<CUDABuffer2DPitch, std::shared_ptr<CUDABuffer2DPitch>>(
          m, "CUDABuffer2DPitch", py::module_local());

  auto _FFmpegAudioFrames =
      py::class_<FFmpegAudioFramesWrapper, FFmpegAudioFramesWrapperPtr>(
          m, "FFmpegAudioFrames", py::module_local());

  auto _FFmpegVideoFrames =
      py::class_<FFmpegVideoFramesWrapper, FFmpegVideoFramesWrapperPtr>(
          m, "FFmpegVideoFrames", py::module_local());

  auto _FFmpegImageFrames =
      py::class_<FFmpegImageFramesWrapper, FFmpegImageFramesWrapperPtr>(
          m, "FFmpegImageFrames", py::module_local());

  auto _NvDecVideoFrames =
      py::class_<NvDecVideoFramesWrapper, NvDecVideoFramesWrapperPtr>(
          m, "NvDecVideoFrames", py::module_local());

  auto _NvDecImageFrames =
      py::class_<NvDecImageFramesWrapper, NvDecImageFramesWrapperPtr>(
          m, "NvDecImageFrames", py::module_local());

  _CPUBuffer
      .def_property_readonly(
          "channel_last",
          [](const CPUBuffer& self) { return self.channel_last; })
      .def_property_readonly(
          "ndim", [](const CPUBuffer& self) { return self.shape.size(); })
      .def_property_readonly(
          "shape", [](const CPUBuffer& self) { return self.shape; })
      .def_property_readonly("is_cuda", [](const CPUBuffer&) { return false; })
      .def_property_readonly("__array_interface__", [](CPUBuffer& self) {
        return get_array_interface(self);
      });

#ifdef SPDL_USE_CUDA
#define IF_CUDABUFFER_ENABLED(x) x
#else
#define IF_CUDABUFFER_ENABLED(x)                                         \
  [](const CUDABuffer&) {                                                \
    throw std::runtime_error("SPDL is not compiled with CUDA support."); \
  }
#endif

  _CUDABuffer
      .def_property_readonly(
          "channel_last", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.channel_last;
          }))
      .def_property_readonly(
          "ndim", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.shape.size();
          }))
      .def_property_readonly(
          "shape", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.shape;
          }))
      .def_property_readonly("is_cuda", [](const CUDABuffer&) { return true; })
      .def_property_readonly(
          "device_index", IF_CUDABUFFER_ENABLED([](const CUDABuffer& self) {
            return self.device_index;
          }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_CUDABUFFER_ENABLED(
              [](CUDABuffer& self) { return get_cuda_array_interface(self); }));

#ifdef SPDL_USE_NVDEC
#define IF_CUDABUFFER2_ENABLED(x) x
#else
#define IF_CUDABUFFER2_ENABLED(x)                                         \
  [](const CUDABuffer2DPitch&) {                                          \
    throw std::runtime_error("SPDL is not compiled with NVDEC support."); \
  }
#endif

  _CUDABuffer2DPitch
      .def_property_readonly(
          "channel_last",
          IF_CUDABUFFER2_ENABLED(
              [](const CUDABuffer2DPitch& self) { return self.channel_last; }))
      .def_property_readonly(
          "ndim", IF_CUDABUFFER2_ENABLED([](const CUDABuffer2DPitch& self) {
            return self.get_shape().size();
          }))
      .def_property_readonly(
          "shape", IF_CUDABUFFER2_ENABLED(&CUDABuffer2DPitch::get_shape))
      .def_property_readonly(
          "is_cuda", IF_CUDABUFFER2_ENABLED([](const CUDABuffer2DPitch& self) {
            return true;
          }))
      .def_property_readonly(
          "device_index",
          IF_CUDABUFFER2_ENABLED(
              [](const CUDABuffer2DPitch& self) { return self.device_index; }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_CUDABUFFER2_ENABLED([](CUDABuffer2DPitch& self) {
            return get_cuda_array_interface(self);
          }));

  _FFmpegAudioFrames
      .def_property_readonly(
          "num_frames",
          [](FFmpegAudioFramesWrapper& self) {
            return self.get_frames_ref()->get_num_frames();
          })
      .def_property_readonly(
          "sample_rate",
          [](FFmpegAudioFramesWrapper& self) {
            return self.get_frames_ref()->get_sample_rate();
          })
      .def_property_readonly(
          "num_channels",
          [](FFmpegAudioFramesWrapper& self) {
            return self.get_frames_ref()->get_num_channels();
          })
      .def_property_readonly(
          "format",
          [](FFmpegAudioFramesWrapper& self) {
            return self.get_frames_ref()->get_media_format_name();
          })
      .def(
          "__len__",
          [](FFmpegAudioFramesWrapper& self) {
            return self.get_frames_ref()->get_num_frames();
          })
      .def("__repr__", [](const FFmpegAudioFramesWrapper& self) {
        auto& ref = self.get_frames_ref();
        return fmt::format(
            "FFmpegAudioFrames<num_frames={}, sample_format=\"{}\", sample_rate={}, num_channels={}, pts={}>",
            ref->get_num_frames(),
            ref->get_media_format_name(),
            ref->get_sample_rate(),
            ref->get_num_channels(),
            double(ref->get_frames().front()->pts) * ref->time_base.num /
                ref->time_base.den);
      });

  _FFmpegVideoFrames
      .def_property_readonly(
          "num_frames",
          [](FFmpegVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_num_frames();
          })
      .def_property_readonly(
          "num_planes",
          [](FFmpegVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_num_planes();
          })
      .def_property_readonly(
          "width",
          [](FFmpegVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_width();
          })
      .def_property_readonly(
          "height",
          [](FFmpegVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_height();
          })
      .def_property_readonly(
          "format",
          [](FFmpegVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_media_format_name();
          })
      .def(
          "__len__",
          [](FFmpegVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_num_frames();
          })
      .def(
          "__getitem__",
          [](const FFmpegVideoFramesWrapper& self, const py::slice& slice) {
            auto& ref = self.get_frames_ref();
            py::ssize_t start = 0, stop = 0, step = 0, len = 0;
            if (!slice.compute(
                    ref->get_num_frames(), &start, &stop, &step, &len)) {
              throw py::error_already_set();
            }
            return wrap<MediaType::Video, FFmpegFramesPtr>(ref->slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step)));
          })
      .def(
          "__getitem__",
          [](const FFmpegVideoFramesWrapper& self, int i) {
            return wrap<MediaType::Image, FFmpegFramesPtr>(
                self.get_frames_ref()->slice(i));
          })
      .def(
          "_get_pts",
          [](const FFmpegVideoFramesWrapper& self) -> std::vector<double> {
            std::vector<double> ret;
            auto& frames = self.get_frames_ref();
            auto base = frames->time_base;
            for (auto& frame : frames->get_frames()) {
              ret.push_back(double(frame->pts) * base.num / base.den);
            }
            return ret;
          })
      .def("__repr__", [](const FFmpegVideoFramesWrapper& self) {
        auto& ref = self.get_frames_ref();
        return fmt::format(
            "FFmpegVideoFrames<num_frames={}, pixel_format=\"{}\", num_planes={}, width={}, height={}, pts={}, is_cuda={}>",
            ref->get_num_frames(),
            ref->get_media_format_name(),
            ref->get_num_planes(),
            ref->get_width(),
            ref->get_height(),
            double(ref->get_frames().front()->pts) * ref->time_base.num /
                ref->time_base.den,
            ref->is_cuda());
      });

  _FFmpegImageFrames
      .def_property_readonly(
          "num_planes",
          [](const FFmpegImageFramesWrapper& self) {
            return self.get_frames_ref()->get_num_planes();
          })
      .def_property_readonly(
          "width",
          [](const FFmpegImageFramesWrapper& self) {
            return self.get_frames_ref()->get_width();
          })
      .def_property_readonly(
          "height",
          [](const FFmpegImageFramesWrapper& self) {
            return self.get_frames_ref()->get_height();
          })
      .def_property_readonly(
          "format",
          [](const FFmpegImageFramesWrapper& self) {
            return self.get_frames_ref()->get_media_format_name();
          })
      .def("__repr__", [](const FFmpegImageFramesWrapper& self) {
        auto& ref = self.get_frames_ref();
        return fmt::format(
            "FFmpegImageFrames<pixel_format=\"{}\", num_planes={}, width={}, height={}, is_cuda={}>",
            ref->get_media_format_name(),
            ref->get_num_planes(),
            ref->get_width(),
            ref->get_height(),
            ref->is_cuda());
      });

#ifdef SPDL_USE_NVDEC
#define IF_NVDECVIDEOFRAMES_ENABLED(x) x
#else
#define IF_NVDECVIDEOFRAMES_ENABLED(x)                                    \
  [](const NvDecVideoFramesWrapper&) {                                    \
    throw std::runtime_error("SPDL is not compiled with NVDEC support."); \
  }
#endif

  // TODO: Add __repr__
  _NvDecVideoFrames
      .def_property_readonly(
          "channel_last",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFramesWrapper& self) {
            return self.get_frames_ref()->buffer->channel_last;
          }))
      .def_property_readonly(
          "ndim",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFramesWrapper& self) {
            return self.get_frames_ref()->buffer->get_shape().size();
          }))
      .def_property_readonly(
          "shape",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFramesWrapper& self) {
            return self.get_frames_ref()->buffer->get_shape();
          }))
      .def_property_readonly(
          "format",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFramesWrapper& self) {
            return self.get_frames_ref()->get_media_format_name();
          }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_NVDECVIDEOFRAMES_ENABLED([](NvDecVideoFramesWrapper& self) {
            return get_cuda_array_interface(*self.get_frames_ref()->buffer);
          }))
      .def(
          "__len__",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecVideoFramesWrapper& self) {
            return self.get_frames_ref()->buffer->get_shape()[0];
          }));

  // TODO: Add __repr__
  _NvDecImageFrames
      .def_property_readonly(
          "channel_last",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFramesWrapper& self) {
            return self.get_frames_ref()->buffer->channel_last;
          }))
      .def_property_readonly(
          "ndim",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFramesWrapper& self) {
            return self.get_frames_ref()->buffer->get_shape().size();
          }))
      .def_property_readonly(
          "shape",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFramesWrapper& self) {
            return self.get_frames_ref()->buffer->get_shape();
          }))
      .def_property_readonly(
          "format",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFramesWrapper& self) {
            return self.get_frames_ref()->get_media_format_name();
          }))
      .def_property_readonly(
          "__cuda_array_interface__",
          IF_NVDECVIDEOFRAMES_ENABLED([](NvDecImageFramesWrapper& self) {
            return get_cuda_array_interface(*self.get_frames_ref()->buffer);
          }))
      .def(
          "__len__",
          IF_NVDECVIDEOFRAMES_ENABLED([](const NvDecImageFramesWrapper& self) {
            return self.get_frames_ref()->buffer->get_shape()[0];
          }));
}
} // namespace spdl::core
