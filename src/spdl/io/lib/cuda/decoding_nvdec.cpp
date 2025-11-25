/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_cuda_extensions.h"

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

#ifndef SPDL_USE_NVCODEC
NvDecDecoder::NvDecDecoder() {}
NvDecDecoder::~NvDecDecoder() {}
void NvDecDecoder::reset() {
  NOT_SUPPORTED_NVCODEC;
}
void NvDecDecoder::init(
    const CUDAConfig&,
    const spdl::core::VideoCodec&,
    CropArea,
    int,
    int) {
  NOT_SUPPORTED_NVCODEC;
}
std::vector<CUDABuffer> NvDecDecoder::decode(spdl::core::VideoPacketsPtr) {
  NOT_SUPPORTED_NVCODEC;
}
std::vector<CUDABuffer> NvDecDecoder::flush() {
  NOT_SUPPORTED_NVCODEC;
}
#endif

using namespace spdl::core;

void register_decoding_nvdec(nb::module_& m) {
  nb::class_<NvDecDecoder>(
      m,
      "NvDecDecoder",
      R"(Decods video packets using NVDEC hardware acceleration.

Use :py:func:`nvdec_decoder` to instantiate.

To decode videos with NVDEC, you provide the decoder configuration
and codec information directly to :py:func:`nvdec_decoder`, then feed
video packets. Finally, call flush to let the decoder know that it
reached the end of the video stream, so that the decoder flushes its
internally buffered frames.

.. note::

   To decode H264 and HEVC videos, the packets must be Annex B
   format. You can convert video packets to Annex B format by
   applying bit stream filter while demuxing or after demuxing.
   See the examples bellow.

.. seealso::

   - :py:func:`decode_packets_nvdec`: Decode video packets using
     NVDEC.
   - :py:func:`streaming_load_video_nvdec`: Decode video frames
     from source in streaming fashion.
   - :py:mod:`streaming_nvdec_decoding`: Demonstrates how to
     decode a long video using NVDEC.

.. admonition:: Example - decoding the whole video

   .. code-block::

      cuda_config = spdl.io.cuda_config(device_index=0)

      packets = spdl.io.demux_video(src)
      # Convert to Annex B format
      if (c := packets.codec.name) in ("h264", "hevc"):
          packets = spdl.io.apply_bsf(f"{c}_mp4toannexb")

      # Initialize the decoder
      decoder = nvdec_decoder(cuda_config, packets.codec)

      # Decode packets
      frames = decoder.decode(packets)

      # Done
      frames += decoder.flush()

      # Convert (and batch) the NV12 frames into RGB
      frames = spdl.io.nv12_to_rgb(frames)

.. admonition:: Example - incremental decoding

   .. code-block::

      cuda_config = spdl.io.cuda_config(device_index=0)

      demuxer = spdl.io.Demuxer(src)
      codec = demuxer.video_codec

      match codec.name:
          case "h264" | "hevc":
              bsf = f"{codec.name}_mp4toannexb"
          case _:
              bsf = None

      # Initialize the decoder
      decoder = nvdec_decoder(cuda_config, codec)

      for packets in demuxer.streaming_demux_video(10, bsf=bsf):
          buffer = decoder.decode(packets)
          buffer = spdl.io.nv12_to_rgb(buffer)
          # Process buffer here

      buffer = decoder.flush()
      buffer = spdl.io.nv12_to_rgb(buffer)
)")
      .def(
          "_reset",
          &NvDecDecoder::reset,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "init",
          [](NvDecDecoder& self,
             const CUDAConfig& cuda_config,
             const spdl::core::VideoCodec& codec,
             int crop_left,
             int crop_top,
             int crop_right,
             int crop_bottom,
             int width,
             int height) {
            self.init(
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
          "decode",
          &NvDecDecoder::decode,
          R"(Decode video frames from the give packets.

.. note::

   Due to how video codec works, the number of returned frames
   do not necessarily match the number of packets provided.

   The method can return less number of frames or more number of
   frames.

Args:
    packets: Video packets.

Returns:
    The decoded frames.
)",
          nb::call_guard<nb::gil_scoped_release>(),
          nb::arg("packets"))
      .def(
          "flush",
          &NvDecDecoder::flush,
          R"(Notify the decoder the end of video stream, and fetch buffered frames.

Returns:
    The decoded frames. (can be empty)
)",
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
