#include <libspdl/core/frames.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace nb = nanobind;

namespace spdl::core {
void register_frames(nb::module_& m) {
  nb::class_<FFmpegAudioFrames>(m, "FFmpegAudioFrames")
      .def_prop_ro(
          "num_frames",
          [](FFmpegAudioFrames& self) {
            nb::gil_scoped_release g;
            return self.get_num_frames();
          })
      .def_prop_ro(
          "sample_rate",
          [](FFmpegAudioFrames& self) {
            nb::gil_scoped_release g;
            return self.get_sample_rate();
          })
      .def_prop_ro(
          "num_channels",
          [](FFmpegAudioFrames& self) {
            nb::gil_scoped_release g;
            return self.get_num_channels();
          })
      .def_prop_ro(
          "format",
          [](FFmpegAudioFrames& self) {
            nb::gil_scoped_release g;
            return self.get_media_format_name();
          })
      .def(
          "__len__",
          [](FFmpegAudioFrames& self) { return self.get_num_frames(); })
      .def(
          "__repr__",
          [](const FFmpegAudioFrames& self) {
            auto num_frames = self.get_num_frames();
            auto pts = [&]() {
              if (num_frames == 0) {
                return std::numeric_limits<double>::quiet_NaN();
              }
              return double(self.get_frames().front()->pts) *
                  self.time_base.num / self.time_base.den;
            }();
            return fmt::format(
                "FFmpegAudioFrames<num_frames={}, sample_format=\"{}\", sample_rate={}, num_channels={}, pts={}>",
                num_frames,
                self.get_media_format_name(),
                self.get_sample_rate(),
                self.get_num_channels(),
                pts);
          })
      .def("clone", [](const FFmpegAudioFrames& self) {
        nb::gil_scoped_release g;
        return clone(self);
      });

  nb::class_<FFmpegVideoFrames>(m, "FFmpegVideoFrames")
      .def_prop_ro(
          "num_frames",
          [](FFmpegVideoFrames& self) {
            nb::gil_scoped_release g;
            return self.get_num_frames();
          })
      .def_prop_ro(
          "num_planes",
          [](FFmpegVideoFrames& self) {
            nb::gil_scoped_release g;
            return self.get_num_planes();
          })
      .def_prop_ro(
          "width",
          [](FFmpegVideoFrames& self) {
            nb::gil_scoped_release g;
            return self.get_width();
          })
      .def_prop_ro(
          "height",
          [](FFmpegVideoFrames& self) {
            nb::gil_scoped_release g;
            return self.get_height();
          })
      .def_prop_ro(
          "format",
          [](FFmpegVideoFrames& self) {
            nb::gil_scoped_release g;
            return self.get_media_format_name();
          })
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
            nb::gil_scoped_release g;
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
            auto num_frames = self.get_num_frames();
            auto pts = [&]() {
              if (num_frames == 0) {
                return std::numeric_limits<double>::quiet_NaN();
              }
              return double(self.get_frames().front()->pts) *
                  self.time_base.num / self.time_base.den;
            }();
            return fmt::format(
                "FFmpegVideoFrames<num_frames={}, pixel_format=\"{}\", num_planes={}, width={}, height={}, pts={}>",
                num_frames,
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height(),
                pts);
          })
      .def("clone", [](const FFmpegVideoFrames& self) {
        nb::gil_scoped_release g;
        return clone(self);
      });

  nb::class_<FFmpegImageFrames>(m, "FFmpegImageFrames")
      .def_prop_ro(
          "num_planes",
          [](const FFmpegImageFrames& self) {
            nb::gil_scoped_release g;
            return self.get_num_planes();
          })
      .def_prop_ro(
          "width",
          [](const FFmpegImageFrames& self) {
            nb::gil_scoped_release g;
            return self.get_width();
          })
      .def_prop_ro(
          "height",
          [](const FFmpegImageFrames& self) {
            nb::gil_scoped_release g;
            return self.get_height();
          })
      .def_prop_ro(
          "format",
          [](const FFmpegImageFrames& self) {
            nb::gil_scoped_release g;
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
      .def(
          "clone",
          [](const FFmpegImageFrames& self) {
            nb::gil_scoped_release g;
            return clone(self);
          })
      .def_prop_ro("pts", [](const FFmpegImageFrames& self) -> double {
        nb::gil_scoped_release g;
        auto base = self.time_base;
        auto& frame = self.get_frames().at(0);
        return double(frame->pts) * base.num / base.den;
      });
}
} // namespace spdl::core
