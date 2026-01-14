/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

#include <libspdl/core/demuxing.h>
#include <libspdl/cuda/nvdec/decoder.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace spdl::cuda {

#define NOT_SUPPORTED_NVCODEC \
  throw std::runtime_error("SPDL is not built with NVCODEC support.")

#ifdef SPDL_USE_NVCODEC
#define _(var_name) var_name
#else
#define _(var_name)

NvDecDecoder::NvDecDecoder() {}
NvDecDecoder::~NvDecDecoder() {}
CUDABuffer NvDecDecoder::decode_packets(spdl::core::VideoPacketsPtr) {
  NOT_SUPPORTED_NVCODEC;
}
void NvDecDecoder::init_decoder(
    const CUDAConfig&,
    const spdl::core::VideoCodec&,
    CropArea,
    int,
    int) {
  NOT_SUPPORTED_NVCODEC;
}
void NvDecDecoder::reset() {
  NOT_SUPPORTED_NVCODEC;
}
void NvDecDecoder::init_buffer(size_t) {
  NOT_SUPPORTED_NVCODEC;
}
#endif

using namespace spdl::core;

void register_decoding_nvdec(nb::module_& m) {
  nb::class_<NvDecDecoder>(
      m,
      "NvDecDecoder",
      R"(Decodes video packets using NVDEC hardware acceleration.

Use :py:func:`nvdec_decoder` to instantiate.

This decoder supports two decoding workflows:

1. **Batch decoding** - Decode all packets at once:

   .. code-block:: python

      decoder = spdl.io.nvdec_decoder(cuda_config, codec)
      nv12_buffer = decoder.decode_packets(packets)
      rgb = spdl.io.nv12_to_rgb(nv12_buffer, cuda_config)
      # Buffer shape: [num_frames, height*1.5, width]

2. **Streaming decoding** - Process packets incrementally with batched output:

   .. code-block:: python

      decoder = spdl.io.nvdec_decoder(cuda_config, codec)
      decoder.init_buffer(num_frames)  # Initialize frame buffer

      for packets in packet_stream:
          for batch in decoder.streaming_decode_packets(packets):
              # batch is CUDABuffer with shape [num_frames, h*1.5, w]
              rgb = spdl.io.nv12_to_rgb(batch, cuda_config)
              # Process rgb frames

      # Flush and get remaining frames
      for batch in decoder.flush():
          rgb = spdl.io.nv12_to_rgb(batch, cuda_config)
          # Process final frames

.. note::

   To decode H264 and HEVC videos, packets must be in Annex B format.
   Use :py:func:`~spdl.io.apply_bsf` to convert:

   .. code-block:: python

      if codec.name in ("h264", "hevc"):
          packets = spdl.io.apply_bsf(packets, f"{codec.name}_mp4toannexb")

.. seealso::

   - :py:func:`decode_packets_nvdec`: Decode video packets using NVDEC
   - :py:func:`streaming_load_video_nvdec`: Decode video frames in streaming fashion
   - :py:mod:`streaming_nvdec_decoding`: Tutorial for streaming video decode
)")
      .def(
          "_reset",
          &NvDecDecoder::reset,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "init_decoder",
          [](NvDecDecoder& self,
             const CUDAConfig& cuda_config,
             const spdl::core::VideoCodec& codec,
             int crop_left,
             int crop_top,
             int crop_right,
             int crop_bottom,
             int width,
             int height) {
            self.init_decoder(
                cuda_config,
                codec,
                CropArea{
                    static_cast<short>(crop_left),
                    static_cast<short>(crop_top),
                    static_cast<short>(crop_right),
                    static_cast<short>(crop_bottom)},
                width,
                height);
          },
          R"(Initialize the decoder.

.. versionchanged:: 0.2.0

   This method was renamed from ``init()``.

.. deprecated:: 0.1.7

   This method was merged with :py:func:`nvdec_decoder()`.
   Pass these parameters directly to initialize the decoder.
   The old pattern of calling ``decoder.init()`` after ``nvdec_decoder()``
   will be removed in a future version.

.. note::

   Creation of underlying decoder object is expensive.
   Typically, it takes about 300ms or more.

   To mitigate this the implementation tries to reuse the decoder.
   This works if the new video uses the same codecs as
   the previous one, and the difference is limited to the
   resolution of the video.

   If you are processing videos of different codecs, then the
   decoder has to be re-created.

Args:
    cuda_config: The device configuration. Specifies the GPU of which
        video decoder chip is used, the CUDA memory allocator and
        CUDA stream used to fetch the result from the decoder engine.

    codec: The information of the source video.

    crop_left, crop_top, crop_right, crop_bottom (int):
        *Optional:* Crop the given number of pixels from each side.

    scale_width, scale_height (int): *Optional:* Resize the frame.
        Resizing is applied after cropping.
)",
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("device_config"),
          nb::arg("codec"),
          nb::kw_only(),
          nb::arg("crop_left") = 0,
          nb::arg("crop_top") = 0,
          nb::arg("crop_right") = 0,
          nb::arg("crop_bottom") = 0,
          nb::arg("scale_width") = -1,
          nb::arg("scale_height") = -1)
      .def(
          "decode_packets",
          &NvDecDecoder::decode_packets,
          R"(Decode all packets and return NV12 buffer.

This method decodes all packets and flushes the decoder in one operation,
and returns the resulting frames as one contiguous memory buffer.

Args:
    packets: Video packets to decode.

Returns:
    A :py:class:`~spdl.io.CUDABuffer` containing NV12 frames with shape
    ``[num_frames, h*1.5, width]``, where ``num_frames`` reflects
    the actual number of decoded frames, which should match
    the number of packets.
)",
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"))
      .def(
          "init_buffer",
          &NvDecDecoder::init_buffer,
          R"(Initialize frame buffer for streaming decode.

This must be called before using :py:meth:`streaming_decode_packets` for streaming decode.

Args:
    num_frames: The number of frames per batch.
)",
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("num_frames"))
      .def(
          "flush",
          [](NvDecDecoder& _(self)) -> std::unique_ptr<CUDABufferGenerator> {
#ifdef SPDL_USE_NVCODEC
            return std::make_unique<CUDABufferGenerator>(self.flush());
#else
            NOT_SUPPORTED_NVCODEC;
#endif
          },
          R"(Flush the decoder and yield remaining batches.

Call this method at the end of video stream to flush the decoder
and retrieve any remaining buffered frames as batches.

Returns:
    Iterator that yields remaining CUDABuffer batches in NV12 format.
)",
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "streaming_decode_packets",
          [](NvDecDecoder& _(self), spdl::core::VideoPacketsPtr _(packets))
              -> std::unique_ptr<CUDABufferGenerator> {
#ifdef SPDL_USE_NVCODEC
            return std::make_unique<CUDABufferGenerator>(
                self.streaming_decode_packets(std::move(packets)));
#else
            NOT_SUPPORTED_NVCODEC;
#endif
          },
          R"(Streaming decode packets and yield batches.

This method decodes packets and yields batches of frames as they become ready.
The frame buffer must be initialized with init_buffer() before calling this.

Args:
    packets: Video packets to decode.

Returns:
    Iterator that yields CUDABuffer batches in NV12 format.
)",
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"));

  // Bind CUDABufferIterator - iterator for CUDABuffer batches
  nb::class_<CUDABufferGenerator>(m, "CUDABufferIterator")
      .def(
          "__iter__",
          [](CUDABufferGenerator& self) -> CUDABufferGenerator& {
            return self;
          },
          nb::rv_policy::reference)
      .def(
          "__next__",
          [](CUDABufferGenerator& self) -> CUDABuffer {
            if (!self) {
              throw nb::stop_iteration();
            }
            return self();
          },
          nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "_nvdec_decoder",
      []() -> std::unique_ptr<NvDecDecoder> {
#ifdef SPDL_USE_NVCODEC
        return std::make_unique<NvDecDecoder>();
#else
        NOT_SUPPORTED_NVCODEC;
#endif
      },
      nb::call_guard<nb::gil_scoped_release>());
}
} // namespace spdl::cuda
