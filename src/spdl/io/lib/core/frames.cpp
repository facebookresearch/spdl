/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

#include <libspdl/core/frames.h>

#include <fmt/core.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
namespace {

template <MediaType media>
std::string get_ts(const Frames<media>& self) {
  auto num_frames = self.get_frames().size();
  if (num_frames == 0) {
    return "nan";
  }
  return fmt::format(
      "[{:.3f}, {:.3f}]",
      self.get_timestamp(0),
      self.get_timestamp(num_frames - 1));
}

std::vector<double> get_timestamps(const VideoFrames& self) {
  std::vector<double> ret;
  auto n = self.get_num_frames();
  ret.reserve(n);
  for (int i = 0; i < n; ++i) {
    ret.push_back(self.get_timestamp(i));
  }
  return ret;
}
std::vector<int64_t> get_pts(const VideoFrames& self) {
  std::vector<int64_t> ret;
  auto n = self.get_num_frames();
  ret.reserve(n);
  for (int i = 0; i < n; ++i) {
    ret.push_back(self.get_pts(i));
  }
  return ret;
}

std::tuple<int, int> get_time_base(const VideoFrames& self) {
  auto tb = self.get_time_base();
  return {tb.num, tb.den};
}

} // namespace
void register_frames(nb::module_& m) {
  nb::class_<AudioFrames>(
      m,
      "AudioFrames",
      "Audio frames.\n\n"
      "See :doc:`/io/packets_frames_concepts` for information about the Frames base concept.")
      .def_prop_ro(
          "num_frames",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_frames();
          },
          "The number of audio frames. Same as ``__len__`` method.\n\n"
          ".. note::\n\n"
          "   In SPDL,\n"
          "   ``The number of samples`` == ``the number of frames`` x ``the number of channels``")
      .def_prop_ro(
          "sample_rate",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_sample_rate();
          },
          "The sample rate of audio.")
      .def_prop_ro(
          "num_channels",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_channels();
          },
          "The number of channels.")
      .def_prop_ro(
          "sample_fmt",
          [](AudioFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          },
          "The name of sample format.\n\n"
          "Possible values are\n\n"
          "- ``\"u8\"`` for unsigned 8-bit integer.\n"
          "- ``\"s16\"``, ``\"s32\"``, ``\"s64\"`` for signed 16-bit, 32-bit and 64-bit integer.\n"
          "- ``\"flt\"``, ``\"dbl\"`` for 32-bit and 64-bit float.\n\n"
          "If the frame is planar format (separate planes for different channels), the\n"
          "name will be suffixed with ``\"p\"``. When converted to buffer, the buffer's shape\n"
          "will be channel-first format ``(channel, num_samples)`` instead of interweaved\n"
          "``(num_samples, channel)``.")
      .def(
          "__len__",
          [](AudioFrames& self) { return self.get_num_frames(); },
          "Returns the number of frames. Same as ``num_frames``.")
      .def(
          "__repr__",
          [](const AudioFrames& self) -> std::string {
            return fmt::format(
                "AudioFrames<num_frames={}, sample_format=\"{}\", sample_rate={}, num_channels={}, timestamp={}>",
                self.get_num_frames(),
                self.get_media_format_name(),
                self.get_sample_rate(),
                self.get_num_channels(),
                get_ts(self));
          })
      .def(
          "clone",
          &AudioFrames::clone,
          nb::call_guard<nb::gil_scoped_release>(),
          "Clone the frames, so that data can be converted to buffer multiple times.\n\n"
          "Returns:\n"
          "    A clone of the frame.");

  nb::class_<VideoFrames>(
      m,
      "VideoFrames",
      "Video frames.\n\n"
      "See :doc:`/io/packets_frames_concepts` for information about the Frames base concept.")
      .def_prop_ro(
          "num_frames",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_frames();
          },
          "The number of video frames. Same as ``__len__`` method.")
      .def_prop_ro(
          "num_planes",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_planes();
          },
          "The number of planes in the each frame.\n\n"
          ".. note::\n\n"
          "   This corresponds to the number of color components, however\n"
          "   it does not always match with the number of color channels when\n"
          "   the frame is converted to buffer/array object.\n\n"
          "   For example, if a video file is YUV format (which is one of the most\n"
          "   common formats, and comprised of different plane sizes), and\n"
          "   color space conversion is disabled during the decoding, then\n"
          "   the resulting frames are converted to buffer as single channel frame\n"
          "   where all the Y, U, V components are packed.\n\n"
          "   SPDL by default converts the color space to RGB, so this is\n"
          "   usually not an issue.")
      .def_prop_ro(
          "width",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_width();
          },
          "The width of video.")
      .def_prop_ro(
          "height",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_height();
          },
          "The height of video.")
      .def_prop_ro(
          "pix_fmt",
          [](VideoFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          },
          "The name of the pixel format.")
      .def(
          "__len__",
          [](VideoFrames& self) { return self.get_num_frames(); },
          "Returns the number of frames. Same as ``num_frames``.")
      .def(
          "__getitem__",
          [](const VideoFrames& self, const nb::slice& slice) {
            auto [start, stop, step, len] =
                slice.compute(self.get_num_frames());
            return self.slice(
                static_cast<int>(start),
                static_cast<int>(stop),
                static_cast<int>(step));
          },
          "Slice frame by key.\n\n"
          "Args:\n"
          "    key: If the key is int type, a single frame is returned as ``ImageFrames``.\n"
          "        If the key is slice type, a new ``VideoFrames`` object pointing the\n"
          "        corresponding frames are returned.\n\n"
          "Returns:\n"
          "    The sliced frame.")
      .def(
          "__getitem__",
          [](const VideoFrames& self, int64_t i) { return self.slice(i); },
          "Slice frame by key.\n\n"
          "Args:\n"
          "    key: If the key is int type, a single frame is returned as ``ImageFrames``.\n"
          "        If the key is slice type, a new ``VideoFrames`` object pointing the\n"
          "        corresponding frames are returned.\n\n"
          "Returns:\n"
          "    The sliced frame.")
      .def(
          "__getitem__",
          [](const VideoFrames& self, const std::vector<int64_t>& idx) {
            return self.slice(idx);
          },
          "Slice frame by key.\n\n"
          "Args:\n"
          "    key: If the key is int type, a single frame is returned as ``ImageFrames``.\n"
          "        If the key is slice type, a new ``VideoFrames`` object pointing the\n"
          "        corresponding frames are returned.\n\n"
          "Returns:\n"
          "    The sliced frame.")
      .def(
          "get_timestamps",
          get_timestamps,
          nb::call_guard<nb::gil_scoped_release>(),
          "Get the timestamp of frames.")
      .def(
          "get_pts",
          get_pts,
          nb::call_guard<nb::gil_scoped_release>(),
          "Get the PTS (Presentation Time Stamp) in timebase unit.")
      .def_prop_ro(
          "time_base",
          get_time_base,
          nb::call_guard<nb::gil_scoped_release>(),
          "Get the time base of PTS.\n\n"
          "The time base is expressed as ``(Numerator, denominator)``.\n\n"
          "PTS (in seconds) == PTS (in timebase unit) * Numerator / Denominator")
      .def(
          "__repr__",
          [](const VideoFrames& self) -> std::string {
            return fmt::format(
                "VideoFrames<num_frames={}, pixel_format=\"{}\", num_planes={}, width={}, height={}, timestamp={}>",
                self.get_num_frames(),
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height(),
                get_ts(self));
          })
      .def(
          "clone",
          &VideoFrames::clone,
          nb::call_guard<nb::gil_scoped_release>(),
          "Clone the frames, so that data can be converted to buffer multiple times.\n\n"
          "Returns:\n"
          "    A clone of the frame.");

  nb::class_<ImageFrames>(
      m,
      "ImageFrames",
      "Image frames.\n\n"
      "See :doc:`/io/packets_frames_concepts` for information about the Frames base concept.")
      .def_prop_ro(
          "num_planes",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_num_planes();
          },
          "The number of planes in the each frame.\n\n"
          "See :py:class:`~spdl.io.VideoFrames` for a caveat.")
      .def_prop_ro(
          "width",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_width();
          },
          "The width of image.")
      .def_prop_ro(
          "height",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_height();
          },
          "The height of image.")
      .def_prop_ro(
          "pix_fmt",
          [](ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_media_format_name();
          },
          "The name of the pixel format.")
      .def_prop_ro(
          "metadata",
          [](const ImageFrames& self) {
            nb::gil_scoped_release __g;
            return self.get_metadata();
          },
          "Metadata attached to the frame.")
      .def(
          "__repr__",
          [](const ImageFrames& self) {
            auto tb = self.get_time_base();
            if (tb.num) {
              return fmt::format(
                  "ImageFrames<pixel_format=\"{}\", num_planes={}, width={}, height={}, timestamp={:.3f}>",
                  self.get_media_format_name(),
                  self.get_num_planes(),
                  self.get_width(),
                  self.get_height(),
                  self.get_timestamp());
            }
            return fmt::format(
                "ImageFrames<pixel_format=\"{}\", num_planes={}, width={}, height={}>",
                self.get_media_format_name(),
                self.get_num_planes(),
                self.get_width(),
                self.get_height());
          })
      .def(
          "clone",
          &ImageFrames::clone,
          nb::call_guard<nb::gil_scoped_release>(),
          "Clone the frames, so that data can be converted to buffer multiple times.\n\n"
          "Returns:\n"
          "    A clone of the frame.")
      .def_prop_ro(
          "pts",
          [](const ImageFrames& self) -> double {
            nb::gil_scoped_release __g;
            return self.get_timestamp();
          },
          "The presentation time stamp of the image in the source video.\n\n"
          "This property is valid only when the ``ImageFrames`` is created from slicing\n"
          ":py:class:`~spdl.io.VideoFrames` object.");
}
} // namespace spdl::core
