# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os

import numpy as np
import spdl.io
from spdl.io import get_abuffer_desc, get_buffer_desc

from ..fixture import get_sample, load_ref_audio, load_ref_video


def test_filter_graph_abuffer_basic():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i sine -c:a pcm_s16le -t 5 sample.wav"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    codec = demuxer.audio_codec
    decoder = spdl.io.Decoder(codec, filter_desc=None)

    filter_desc = f"{get_abuffer_desc(codec)},anull,abuffersink"
    print(filter_desc)

    filter_graph = spdl.io.FilterGraph(filter_desc)
    print(filter_graph)
    buffers = []
    for packets in demuxer.streaming_demux(duration=1):
        frames = decoder.decode(packets)
        filter_graph.add_frames(frames)
        frames = filter_graph.get_frames()
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    if (frames := decoder.flush()) is not None:
        filter_graph.add_frames(frames)

    filter_graph.flush()

    if (frames := filter_graph.get_frames()) is not None:
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    ref = load_ref_audio(
        sample.path, shape=[-1, 1], filter_desc=None, format="s16le", dtype=np.int16
    )
    hyp = np.concatenate(buffers)

    np.testing.assert_array_equal(hyp, ref)


def test_filter_graph_buffer_basic():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc2 -t 5 sample.mp4"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    codec = demuxer.video_codec
    decoder = spdl.io.Decoder(codec, filter_desc=None)

    filter_desc = f"{get_buffer_desc(codec)},null,buffersink"
    print(filter_desc)

    filter_graph = spdl.io.FilterGraph(filter_desc)
    print(filter_graph)
    buffers = []
    for packets in demuxer.streaming_demux(duration=1):
        frames = decoder.decode(packets)
        filter_graph.add_frames(frames)
        frames = filter_graph.get_frames()

        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    if (frames := decoder.flush()) is not None:
        filter_graph.add_frames(frames)

    filter_graph.flush()

    if (frames := filter_graph.get_frames()) is not None:
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    ref = load_ref_video(
        sample.path,
        shape=[-1, 1, 360, 320],
        filter_desc=None,
    )
    hyp = np.concatenate(buffers)

    np.testing.assert_array_equal(hyp, ref)


def test_filter_graph_multiple_inputs():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc2 -t 5 sample.mp4"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    codec = demuxer.video_codec
    decoder = spdl.io.Decoder(codec, filter_desc=None)

    buf0 = get_buffer_desc(codec, label="in0")
    buf1 = get_buffer_desc(codec, label="in1")
    filter_desc = f"{buf0} [in0];{buf1} [in1],[in0] [in1] vstack,buffersink"
    print(filter_desc)

    filter_graph = spdl.io.FilterGraph(filter_desc)
    print(filter_graph)

    buffers = []
    num_packets = 0
    for packets in demuxer.streaming_demux(duration=1):
        num_packets += len(packets)
        frames = decoder.decode(packets)
        filter_graph.add_frames(frames.clone(), key="buffer@in0")
        filter_graph.add_frames(frames, key="buffer@in1")
        frames = filter_graph.get_frames()

        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    if (frames := decoder.flush()) is not None:
        filter_graph.add_frames(frames.clone(), key="buffer@in0")
        filter_graph.add_frames(frames, key="buffer@in1")

    filter_graph.flush()

    if (frames := filter_graph.get_frames()) is not None:
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    print(f"{num_packets=}")
    ref = load_ref_video(
        sample.path,
        shape=[-1, 1, 720, 320],
        filter_desc=None,
        filter_complex="split [o0][o1];[o0] [o1] vstack",
    )
    hyp = np.concatenate(buffers)

    np.testing.assert_array_equal(hyp, ref)


def test_filter_graph_multiple_outputs():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc2 -t 5 sample.mp4"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    codec = demuxer.video_codec
    decoder = spdl.io.Decoder(codec, filter_desc=None)

    filter_desc = ";".join(
        [
            f"{get_buffer_desc(codec)} [in]",
            "[in] split [out0][out1]",
            "[out0] buffersink@out0",
            "[out1] buffersink@out1",
        ]
    )
    print(filter_desc)

    filter_graph = spdl.io.FilterGraph(filter_desc)
    print(filter_graph)
    buffers0, buffers1 = [], []
    for packets in demuxer.streaming_demux(duration=1):
        frames = decoder.decode(packets)
        filter_graph.add_frames(frames)

        frames = filter_graph.get_frames(key="buffersink@out0")
        buffer = spdl.io.convert_frames(frames)
        buffers0.append(spdl.io.to_numpy(buffer))

        frames = filter_graph.get_frames(key="buffersink@out1")
        buffer = spdl.io.convert_frames(frames)
        buffers1.append(spdl.io.to_numpy(buffer))

    if (frames := decoder.flush()) is not None:
        filter_graph.add_frames(frames)

    filter_graph.flush()

    if (frames := filter_graph.get_frames(key="buffersink@out0")) is not None:
        buffer = spdl.io.convert_frames(frames)
        buffers0.append(spdl.io.to_numpy(buffer))

    if (frames := filter_graph.get_frames(key="buffersink@out1")) is not None:
        buffer = spdl.io.convert_frames(frames)
        buffers1.append(spdl.io.to_numpy(buffer))

    ref = load_ref_video(
        sample.path,
        shape=[-1, 1, 360, 320],
        filter_desc=None,
    )
    hyp0 = np.concatenate(buffers0)
    hyp1 = np.concatenate(buffers1)

    np.testing.assert_array_equal(hyp0, ref)
    np.testing.assert_array_equal(hyp1, ref)


def test_filter_graph_audio_in_video_out():
    cmd = "ffmpeg -hide_banner -y -f lavfi -i sine -c:a pcm_s16le -t 5 sample.wav"

    sample = get_sample(cmd)

    demuxer = spdl.io.Demuxer(sample.path)
    codec = demuxer.audio_codec
    decoder = spdl.io.Decoder(codec, filter_desc=None)

    filter_desc = f"{get_abuffer_desc(codec)},showwaves,buffersink"
    print(filter_desc)

    filter_graph = spdl.io.FilterGraph(filter_desc)
    print(filter_graph)
    buffers = []
    for packets in demuxer.streaming_demux(duration=1):
        print(packets)
        frames = decoder.decode(packets)
        print(frames)
        filter_graph.add_frames(frames)
        frames = filter_graph.get_frames()
        print(frames)
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    if (frames := decoder.flush()) is not None:
        print(frames)
        filter_graph.add_frames(frames)

    filter_graph.flush()

    if (frames := filter_graph.get_frames()) is not None:
        print(frames)
        buffer = spdl.io.convert_frames(frames)
        buffers.append(spdl.io.to_numpy(buffer))

    # The result matches as long as the library and CLI version match.
    # Unfortunately, that's not the case for some of our environment

    if os.environ.get("SPDL_SKIP_FILTER_PARITY_TEST", "0") == "1":
        return

    ref = load_ref_video(
        sample.path,
        filter_desc=None,
        shape=[-1, 240, 600, 4],
        filter_complex="showwaves",
    )
    hyp = np.concatenate(buffers)

    # np.savez("debug.npz", ref=ref, hyp=hyp)
    np.testing.assert_array_equal(hyp, ref[-len(hyp) :])
