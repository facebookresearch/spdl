#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example shows how to process video in streaming fashion.

For the resulting video to be playable, audio data and video data must be
written in small chunks in alternating manner.

.. include:: ../plots/streaming_video_processing_block.txt

The following diagram illustrates how audio/video data are processed.

.. include:: ../plots/streaming_video_processing_chart.txt

We use :py:class:`spdl.io.Demuxer` to extract audio/video data from the
source. (1)

In this example, we do not modify audio data, so audio packets are sent to
muxer (an instance of :py:class:`spdl.io.Muxer`) directly. (2)

To modify video data, first we decode videos packets and obtain frames,
using :py:class:`spdl.io.VideoDecoder`. (3)

Usually, video frames are stored as YUV420 format, so we convert
it to RGB using :py:class:`spdl.io.FilterGraph`. (4) Then the resulting
frame data are extracted as NumPy array. (5)

Though omitted in this example, let's pretend that the array data is
modified with some sort of AI model. Now we convert the array back
to packet, by applying a reverse operation one by one.

To convert array back to frames, we use
:py:func:`spdl.io.create_reference_video_frame`. This function creates a
:py:class:`~spdl.io.VideoFrames` object that references the data of the
array. (6)

We convert RGB into YUV420 using another :py:class:`~spdl.io.FilterGraph`
instance. (7)

The YUV frame is encoded using :py:class:`spdl.io.VideoEncoder`. (8)

Finally, the encoded data is written to the multiplexer. (9)

.. admonition:: Note on component states
   :class: note

   All the media processing components used in this example,
   (Demuxer/Decoder/FilterGraph/Encoder/Muxer) maintain its own internal
   state, and do not necessarily process the input data immediately.

   Therefore, the number of input/output frames/packets do not necessarily
   match, and you need to call ``flush()`` at the end for each component.

"""

__all__ = [
    "main",
    "parse_args",
    "get_filter_desc",
    "process",
    "build_components",
    "main",
]

import argparse
from pathlib import Path

import spdl.io
from spdl.io import (
    Demuxer,
    FilterGraph,
    Muxer,
    VideoDecoder,
    VideoEncoder,
    VideoPackets,
)

# pyre-strict


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--input-path", "-i", required=True, type=Path)
    parser.add_argument("--output-path", "-o", required=True, type=Path)
    return parser.parse_args()


def get_filter_desc(
    input_pix_fmt: str,
    input_width: int,
    input_height: int,
    frame_rate: tuple[int, int],
    output_pix_fmt: str,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Build a filter description that performs format conversion and optional scaling

    Args:
        input_pix_fmt: The input pixel format. Usually ``"rgb24"``.
        input_width,input_height: The input frame resolution.
        frame_rate: The frame rate of the video.
        output_pix_fmt: The output pixel format. It is the pixel format used by
            the encoder.
        output_width,output_height: The output frame resolution.

    Returns:
        The filter description.
    """
    # filter graph for converting RGB into YUV420p
    buffer_arg = ":".join(
        [
            f"video_size={input_width}x{input_height}",
            f"pix_fmt={input_pix_fmt}",
            f"time_base={frame_rate[1]}/{frame_rate[0]}",
            "pixel_aspect=1/1",
        ]
    )
    filter_arg = ",".join(
        [
            f"format=pix_fmts={output_pix_fmt}",
            f"scale=w={output_width or 'iw'}:h={output_height or 'ih'}",
        ]
    )
    return f"buffer={buffer_arg},{filter_arg},buffersink"


def process(
    demuxer: Demuxer,
    video_decoder: VideoDecoder,
    filter_graph: FilterGraph,
    video_encoder: VideoEncoder,
    muxer: Muxer,
) -> None:
    """The main processing logic.

    Args:
        demuxer: Demux audio/video streams from the source.
        video_decoder: Decode the video packets.
        filter_graph: Transform applied to the array data before encoding.
        video_encoder: Encode the processed video array.
        muxer: Multiplexer for remuxing audio packets and processed video packets.
    """
    src_pix_fmt = "rgb24"
    frame_rate = demuxer.video_codec.frame_rate
    video_index = demuxer.video_stream_index
    audio_index = demuxer.audio_stream_index

    streaming_demuxing = demuxer.streaming_demux([video_index, audio_index], duration=1)
    with muxer.open():
        num_video_frames = 0
        for packets in streaming_demuxing:
            if (audio_packets := packets.get(audio_index)) is not None:
                muxer.write(1, audio_packets)

            if (video_packets := packets.get(video_index)) is None:
                continue

            assert isinstance(video_packets, VideoPackets)
            if (frames := video_decoder.decode(video_packets)) is not None:
                buffer = spdl.io.convert_frames(frames)
                array = spdl.io.to_numpy(buffer)

                ##############################################################
                # <ADD FRAME PROCESSING HERE>
                ##############################################################

                frames = spdl.io.create_reference_video_frame(
                    array,
                    pix_fmt=src_pix_fmt,
                    frame_rate=frame_rate,
                    pts=num_video_frames,
                )
                num_video_frames += len(array)

                filter_graph.add_frames(frames)

                if (frames := filter_graph.get_frames()) is not None:
                    if (
                        packets := video_encoder.encode(frames)  # pyre-ignore
                    ) is not None:
                        muxer.write(0, packets)

        # -------------------------------------------------------------
        # Drain mode
        # -------------------------------------------------------------

        # Flush decoder
        if (frames := video_decoder.flush()) is not None:
            buffer = spdl.io.convert_frames(frames)
            array = spdl.io.to_numpy(buffer)

            ##############################################################
            # <ADD FRAME PROCESSING HERE>
            ##############################################################

            frames = spdl.io.create_reference_video_frame(
                array,
                pix_fmt=src_pix_fmt,
                frame_rate=frame_rate,
                pts=num_video_frames,
            )
            num_video_frames += len(frames)

            filter_graph.add_frames(frames)
            if (frames := filter_graph.get_frames()) is not None:
                if (packets := video_encoder.encode(frames)) is not None:  # pyre-ignore
                    muxer.write(0, packets)

        # Flush filter graph
        if (frames := filter_graph.flush()) is not None:
            if (packets := video_encoder.encode(frames)) is not None:
                muxer.write(0, packets)

        # Flush encoder
        if (packets := video_encoder.flush()) is not None:
            muxer.write(0, packets)


def build_components(
    input_path: Path, output_path: Path
) -> tuple[Demuxer, VideoDecoder, FilterGraph, VideoEncoder, Muxer]:
    """"""
    demuxer = spdl.io.Demuxer(input_path)
    muxer = spdl.io.Muxer(output_path)

    # Fetch the input config
    audio_codec = demuxer.audio_codec

    video_codec = demuxer.video_codec
    frame_rate = video_codec.frame_rate
    src_width = video_codec.width
    src_height = video_codec.height

    # Create decoder
    video_decoder = spdl.io.Decoder(demuxer.video_codec)

    # Configure output
    src_pix_fmt = "rgb24"
    enc_pix_fmt = "yuv420p"
    enc_height = src_height // 2
    enc_width = src_width // 2
    filter_desc = get_filter_desc(
        input_pix_fmt=src_pix_fmt,
        input_width=src_width,
        input_height=src_height,
        frame_rate=frame_rate,
        output_pix_fmt=enc_pix_fmt,
        output_width=enc_width,
        output_height=enc_height,
    )
    print(filter_desc)
    filter_graph = spdl.io.FilterGraph(filter_desc)

    video_encoder = muxer.add_encode_stream(
        config=spdl.io.video_encode_config(
            pix_fmt=enc_pix_fmt,
            frame_rate=frame_rate,
            height=enc_height,
            width=enc_width,
            colorspace="bt709",
            color_primaries="bt709",
            color_trc="bt709",
        ),
    )
    muxer.add_remux_stream(audio_codec)
    return demuxer, video_decoder, filter_graph, video_encoder, muxer


def main() -> None:
    """Entrypoint from the command line."""
    args = parse_args()

    demuxer, video_decoder, filter_graph, video_encoder, muxer = build_components(
        args.input_path, args.output_path
    )

    process(
        demuxer,
        video_decoder,
        filter_graph,
        video_encoder,
        muxer,
    )


if __name__ == "__main__":
    main()
