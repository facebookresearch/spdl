/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

#include <libspdl/core/filter_graph.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::core {
void register_filtering(nb::module_& m) {
  nb::class_<FilterGraph>(m, "FilterGraph", R"(Construct a filter graph

Args:
    filter_desc: A filter graph description.

.. seealso::

   - :py:func:`get_buffer_desc`, :py:func:`get_abuffer_desc`: Helper functions
     for constructing input audio/video frames.

.. admonition:: Example - Audio filtering (passthrough)

   For video processing use ``abuffer`` for input and ``abuffersink`` for output.

   .. code-block::

      filter_desc = "abuffer=time_base=1/44100:sample_rate=44100:sample_fmt=s16:channel_layout=1c,anull,abuffersink"
      filter_graph = FilterGraph(filter_desc)

      filter_graph.add_frames(frames)
      frames = filter_graph.get_frames()

   .. code-block::

      +------------------+
      | Parsed_abuffer_0 |default--[44100Hz s16:mono]--Parsed_anull_1:default
      |    (abuffer)     |
      +------------------+

                                                            +----------------+
      Parsed_abuffer_0:default--[44100Hz s16:mono]--default| Parsed_anull_1 |default--[44100Hz s16:mono]--Parsed_abuffersink_2:default
                                                            |    (anull)     |
                                                            +----------------+

                                                          +----------------------+
      Parsed_anull_1:default--[44100Hz s16:mono]--default| Parsed_abuffersink_2 |
                                                          |    (abuffersink)     |
                                                          +----------------------+

.. admonition:: Example - Video filtering (passthrough)

   For video processing use ``buffer`` for input and ``buffersink`` for output.

   .. code-block::

      filter_desc = "buffer=video_size=320x240:pix_fmt=yuv420p:time_base=1/12800:pixel_aspect=1/1,null,buffersink"
      filter_graph = FilterGraph(filter_desc)

      filter_graph.add_frames(frames)
      frames = filter_graph.get_frames()

   .. code-block::

      +-----------------+
      | Parsed_buffer_0 |default--[320x240 1:1 yuv420p]--Parsed_null_1:default
      |    (buffer)     |
      +-----------------+

                                                              +---------------+
      Parsed_buffer_0:default--[320x240 1:1 yuv420p]--default| Parsed_null_1 |default--[320x240 1:1 yuv420p]--Parsed_buffersink_2:default
                                                              |    (null)     |
                                                              +---------------+

                                                            +---------------------+
      Parsed_null_1:default--[320x240 1:1 yuv420p]--default| Parsed_buffersink_2 |
                                                            |    (buffersink)     |
                                                            +---------------------+

.. admonition:: Example - Multiple Inputs

   Suffix the ``buffer``/``abuffer`` with node name so that it can be referred later.

   .. code-block::

      filter_desc = "buffer@in0=video_size=320x240:pix_fmt=yuv420p:time_base=1/12800:pixel_aspect=1/1 [in0];buffer@in1=video_size=320x240:pix_fmt=yuv420p:time_base=1/12800:pixel_aspect=1/1 [in1],[in0] [in1] vstack,buffersink"
      filter_graph = FilterGraph(filter_desc)

      filter_graph.add_frames(frames0, key="buffer@in0")
      filter_graph.add_frames(frames1, key="buffer@in1")
      frames = filter_graph.get_frames()

   .. code-block::

      +------------+
      | buffer@in0 |default--[320x240 1:1 yuv420p]--Parsed_vstack_2:input0
      |  (buffer)  |
      +------------+

      +------------+
      | buffer@in1 |default--[320x240 1:1 yuv420p]--Parsed_vstack_2:input1
      |  (buffer)  |
      +------------+

                                                       +-----------------+
      buffer@in0:default--[320x240 1:1 yuv420p]--input0| Parsed_vstack_2 |default--[320x480 1:1 yuv420p]--Parsed_buffersink_3:default
      buffer@in1:default--[320x240 1:1 yuv420p]--input1|    (vstack)     |
                                                       +-----------------+

                                                              +---------------------+
      Parsed_vstack_2:default--[320x480 1:1 yuv420p]--default| Parsed_buffersink_3 |
                                                              |    (buffersink)     |
                                                              +---------------------+

.. admonition:: Example - Multiple outputs

   Suffix the ``buffersink``/``abuffersink`` with node name so that it can be referred later.

   .. code-block::

      filter_desc = "buffer=video_size=320x240:pix_fmt=yuv420p:time_base=1/12800:pixel_aspect=1/1 [in];[in] split [out0][out1];[out0] buffersink@out0;[out1] buffersink@out1"
      filter_graph = FilterGraph(filter_desc)

      filter_graph.add_frames(frames)
      frames0 = filter_graph.get_frames(key="buffersink@out0")
      frames1 = filter_graph.get_frames(key="buffersink@out1")

   .. code-block::

      +-----------------+
      | Parsed_buffer_0 |default--[320x240 1:1 yuv420p]--Parsed_split_1:default
      |    (buffer)     |
      +-----------------+

                                                              +----------------+
      Parsed_buffer_0:default--[320x240 1:1 yuv420p]--default| Parsed_split_1 |output0--[320x240 1:1 yuv420p]--buffersink@out0:default
                                                              |    (split)     |output1--[320x240 1:1 yuv420p]--buffersink@out1:default
                                                              +----------------+

                                                             +-----------------+
      Parsed_split_1:output0--[320x240 1:1 yuv420p]--default| buffersink@out0 |
                                                             |  (buffersink)   |
                                                             +-----------------+

                                                             +-----------------+
      Parsed_split_1:output1--[320x240 1:1 yuv420p]--default| buffersink@out1 |
                                                             |  (buffersink)   |
                                                             +-----------------+


.. admonition:: Example - Multimedia filter

   Using `multimedia filters <https://ffmpeg.org/ffmpeg-filters.html#Multimedia-Filters>`_
   allows to convert audio stream to video stream.

   .. code-block::

      filter_desc = "abuffer=time_base=1/44100:sample_rate=44100:sample_fmt=s16:channel_layout=1c,showwaves,buffersink"
      filter_graph = FilterGraph(filter_desc)

      filter_graph.add_frames(audio_frames)
      video_frames = filter_graph.get_frames()

   .. code-block::

      +------------------+
      | Parsed_abuffer_0 |default--[44100Hz s16:mono]--Parsed_showwaves_1:default
      |    (abuffer)     |
      +------------------+

                                                            +--------------------+
      Parsed_abuffer_0:default--[44100Hz s16:mono]--default| Parsed_showwaves_1 |default--[600x240 1:1 rgba]--Parsed_buffersink_2:default
                                                            |    (showwaves)     |
                                                            +--------------------+

                                                              +---------------------+
      Parsed_showwaves_1:default--[600x240 1:1 rgba]--default| Parsed_buffersink_2 |
                                                              |    (buffersink)     |
                                                              +---------------------+

See Also:
    - :doc:`../io/advanced_filtering` - Complete guide to complex filter graphs
    - :doc:`../io/filtering` - Basic filter usage
)")
      .def(nb::init<const std::string&>(), nb::arg("filter_desc"))
      .def(
          "add_frames",
          &FilterGraph::add_frames,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("frames"),
          nb::kw_only(),
          nb::arg("key") = std::nullopt,
          R"(Add a frame to an input node of the filter graph.

Args:
    frames: An input frames object.
    key: The name of the input node.
        This is required when the graph has multiple input nodes.
)")
      .def(
          "flush",
          &FilterGraph::flush,
          nb::call_guard<nb::gil_scoped_release>(),
          R"(Notify the graph that all the input stream reached the end.)")
      .def(
          "get_frames",
          &FilterGraph::get_frames,
          nb::call_guard<nb::gil_scoped_release>(),
          nb::kw_only(),
          nb::arg("key") = std::nullopt,
          R"(Get a frame from an output node of the filter graph.

Args:
    key: The name of the output node.
        This is required when the graph has multiple output nodes.

Returns:
    A Frames object if an output is ready, otherwise ``None``.
)")
      .def(
          "dump", &FilterGraph::dump, nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "make_filter_graph",
      &make_filter_graph,
      nb::call_guard<nb::gil_scoped_release>(),
      nb::arg("filter_desc"));
}
} // namespace spdl::core
