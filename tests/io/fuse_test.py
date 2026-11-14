# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import contextlib
import dataclasses
import inspect
import pickle
import unittest
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import spdl.io
import torch
from spdl.io import _core, _fuse, _preprocessing
from spdl.io._fuse import fuse
from spdl.io.lib import _libspdl

from ..fixture import FFMPEG_CLI, get_sample, get_samples, SrcInfo

_VIDEO_CMD = (
    f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=size=64x48 "
    "-f lavfi -i sine -t 3 sample.mp4"
)
_GOP_VIDEO_CMD = (
    f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=size=64x48 "
    '-force_key_frames "expr:eq(mod(n, 25), 0)" -frames:v 200 sample.mp4'
)
_AUDIO_CMD = (
    f"{FFMPEG_CLI} -hide_banner -y -f lavfi "
    "-i sine=frequency=1000:sample_rate=16000:duration=2 -c:a pcm_s16le sample.wav"
)


def _image_samples(n: int) -> list[SrcInfo]:
    cmd = (
        f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc=size=64x48 "
        f"-frames:v {n} sample_%03d.png"
    )
    return sorted(get_samples(cmd), key=lambda s: s.path)


def _read_sample(cmd: str) -> bytes:
    sample = get_sample(cmd)
    return Path(sample.path).read_bytes()


def _read(samples: list[SrcInfo]) -> list[bytes]:
    return [Path(s.path).read_bytes() for s in samples]


def _normalize(obj: object) -> object:
    """Convert results so that eager and fused outputs can be compared."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_normalize(v) for v in obj)
    if hasattr(obj, "__array_interface__") and not isinstance(obj, np.ndarray):
        return np.array(spdl.io.to_numpy(obj))  # pyrefly: ignore[bad-argument-type]
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    return obj


def setUpModule() -> None:
    # `fuse` does nothing without the GIL, or when it is disabled by the switch
    # read at decoration. Test recording on any Python build, regardless of
    # the switch.
    for name, value in (("_gil_enabled", True), ("_is_fusion_disabled", False)):
        patcher = mock.patch.object(_fuse, name, return_value=value)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


_LOADERS = (
    spdl.io.load_audio,
    spdl.io.load_video,
    spdl.io.load_image,
    spdl.io.load_image_batch,
)


@contextlib.contextmanager
def _eager() -> Iterator[None]:
    """Run without fusion, for reference results.

    The loading functions of `spdl.io` are fused themselves, so a reference
    that calls them must disable fusion to be eager. `fuse` does nothing
    without the GIL, which is checked on every call.
    """
    with mock.patch.object(_fuse, "_gil_enabled", return_value=False):
        yield


class _FuseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # The loaders cache programs across calls. Start each test fresh,
        # so that the number of native calls does not depend on test order.
        for loader in _LOADERS:
            loader._clear_cache()  # pyrefly: ignore[missing-attribute]

    def assert_same(self, hyp: object, ref: object) -> None:
        hyp, ref = _normalize(hyp), _normalize(ref)
        if isinstance(ref, dict):
            assert isinstance(hyp, dict)
            self.assertEqual(hyp.keys(), ref.keys())
            for k in ref:
                self.assert_same(hyp[k], ref[k])
        elif isinstance(ref, (list, tuple)):
            assert isinstance(hyp, (list, tuple))
            self.assertEqual(len(hyp), len(ref))
            for h, r in zip(hyp, ref, strict=True):
                self.assert_same(h, r)
        elif isinstance(ref, np.ndarray):
            np.testing.assert_array_equal(hyp, ref, strict=True)
        else:
            self.assertEqual(hyp, ref)

    def assert_fused_matches_eager(
        self, fn: Callable[..., Any], *args: object
    ) -> object:
        """Run ``fn`` eagerly and fused (recorded, traced and cached), and compare."""
        with _eager():
            ref = fn(*args)
        fused = fuse(fn)
        for _ in range(3):
            self.assert_same(fused(*args), ref)
        return ref


def _eager_image_batch(srcs: list[bytes]) -> np.ndarray:
    frames = [spdl.io.decode_packets(spdl.io.demux_image(s)) for s in srcs]
    return spdl.io.to_numpy(spdl.io.convert_frames(frames))


@fuse
def _fused_image_batch(srcs: Sequence[str | bytes]) -> spdl.io.CPUBuffer:
    frames = [spdl.io.decode_packets(spdl.io.demux_image(s)) for s in srcs]
    return spdl.io.convert_frames(frames)


class FuseImageTest(_FuseTestCase):
    def test_list_comprehension_matches_eager(self) -> None:
        """A batch built with list comprehension matches the eager result."""
        samples = _image_samples(3)
        data = _read(samples)
        paths = [s.path for s in samples]

        ref = _eager_image_batch(data)
        for srcs in (data, paths):
            with self.subTest(src_type=type(srcs[0])):
                hyp = spdl.io.to_numpy(_fused_image_batch(srcs))
                np.testing.assert_array_equal(hyp, ref, strict=True)

    def test_fused_call_does_not_call_eager_bindings(self) -> None:
        """Tracing and running the fused function do not go through per-op bindings."""
        data = _read(_image_samples(2))

        def load(srcs: list[bytes]) -> spdl.io.CPUBuffer:
            frames = [
                spdl.io.decode_packets(
                    spdl.io.demux_image(s), filter_desc="scale=32:24,format=gray"
                )
                for s in srcs
            ]
            return spdl.io.convert_frames(frames)

        ref = spdl.io.to_numpy(load(data))
        fused = fuse(load)
        with mock.patch.object(_core, "_libspdl", new=mock.NonCallableMock(spec=[])):
            results = [fused(data) for _ in range(3)]

        for result in results:
            np.testing.assert_array_equal(spdl.io.to_numpy(result), ref, strict=True)

    def test_retrace_per_structure(self) -> None:
        """A different number of sources produces a differently-shaped batch."""
        data = _read(_image_samples(3))
        self.assertEqual(spdl.io.to_numpy(_fused_image_batch(data[:2])).shape[0], 2)
        self.assertEqual(spdl.io.to_numpy(_fused_image_batch(data)).shape[0], 3)

    def test_nested_outputs(self) -> None:
        """Multiple buffers can be returned in a container."""
        data = _read(_image_samples(2))

        def load(srcs: list[bytes]) -> dict[str, list[spdl.io.CPUBuffer]]:
            return {
                "images": [
                    spdl.io.convert_frames(
                        spdl.io.decode_packets(spdl.io.demux_image(s))
                    )
                    for s in srcs
                ]
            }

        self.assert_fused_matches_eager(load, data)

    def test_image_metadata(self) -> None:
        """Image packets, codec and frames metadata match eager."""
        data = _read(_image_samples(1))[0]

        def load(src: bytes) -> dict[str, object]:
            packets = spdl.io.demux_image(src)
            codec = packets.codec
            assert codec is not None
            meta: dict[str, object] = {
                "packets": (packets.width, packets.height, packets.pix_fmt),
                "codec": (codec.name, codec.time_base),
            }
            frames = spdl.io.decode_packets(packets)
            meta["frames"] = (frames.width, frames.height, frames.pix_fmt)
            return {"meta": meta, "image": spdl.io.convert_frames(frames)}

        self.assert_fused_matches_eager(load, data)

    def test_composite_loaders(self) -> None:
        """`load_image` and `load_image_batch` can be fused."""
        data = _read(_image_samples(3))

        def load(srcs: list[bytes]) -> tuple[object, object]:
            return (
                spdl.io.load_image(srcs[0]),
                spdl.io.load_image_batch(srcs, width=32, height=24),
            )

        self.assert_fused_matches_eager(load, data)

    def test_default_filter_resolving_to_none(self) -> None:
        """A default filter that resolves to `None` disables filtering."""
        data = _read(_image_samples(1))[0]

        def run(src: bytes) -> object:
            return spdl.io.load_image(src)

        unfiltered = spdl.io.load_image(data, filter_desc=None)
        with mock.patch.object(
            _preprocessing, "get_video_filter_desc", return_value=None
        ):
            ref = self.assert_fused_matches_eager(run, data)
        self.assert_same(ref, unfiltered)

    def test_array_conversion(self) -> None:
        """`to_numpy` and `to_torch` are applied to the fused result."""
        data = _read(_image_samples(1))[0]

        @fuse
        def load(src: bytes) -> tuple[object, object]:
            return (
                spdl.io.to_numpy(spdl.io.load_image(src)),
                spdl.io.to_torch(spdl.io.load_image(src)),
            )

        array, tensor = load(data)
        self.assertIsInstance(array, np.ndarray)
        self.assertIsInstance(tensor, torch.Tensor)
        np.testing.assert_array_equal(tensor.numpy(), array, strict=True)
        with _eager():
            ref = spdl.io.to_numpy(spdl.io.load_image(data))
        np.testing.assert_array_equal(array, ref, strict=True)


class FuseVideoTest(_FuseTestCase):
    def test_video_and_audio_with_timestamp(self) -> None:
        """Video and audio demuxed from one source with windows match eager."""
        data = _read_sample(_VIDEO_CMD)
        windows = [(0.0, 1.0), (1.0, 2.0)]

        def run(src: bytes) -> tuple[spdl.io.CPUBuffer, spdl.io.CPUBuffer]:
            video = spdl.io.convert_frames(
                [
                    spdl.io.decode_packets(spdl.io.demux_video(src, timestamp=w))
                    for w in windows
                ]
            )
            audio = spdl.io.convert_frames(
                spdl.io.decode_packets(spdl.io.demux_audio(src, timestamp=(0, 1)))
            )
            return video, audio

        self.assert_fused_matches_eager(run, data)

    def test_demuxer_multiple_windows_and_metadata(self) -> None:
        """One `Demuxer` serves multiple windows, and metadata matches eager."""
        data = _read_sample(_VIDEO_CMD)
        windows = [(0.0, 0.5), (1.0, 1.5)]

        def run(src: bytes) -> dict[str, object]:
            with spdl.io.Demuxer(src) as demuxer:
                codec = demuxer.video_codec
                meta: dict[str, object] = {
                    "codec": (
                        codec.name,
                        codec.width,
                        codec.height,
                        codec.pix_fmt,
                        codec.frame_rate,
                        codec.time_base,
                        codec.sample_aspect_ratio,
                    ),
                    "audio_codec": demuxer.audio_codec.sample_rate,
                    "has_audio": demuxer.has_audio(),
                    "stream_index": demuxer.video_stream_index,
                }
                packets = [demuxer.demux_video(w) for w in windows]
            meta["packets"] = [
                (
                    p.timestamp,
                    p.width,
                    p.height,
                    p.pix_fmt,
                    p.frame_rate,
                    p.get_timestamps(),
                    p.get_timestamps(raw=True),
                )
                for p in packets
            ]
            frames = [spdl.io.decode_packets(p) for p in packets]
            meta["frames"] = [
                (
                    f.num_frames,
                    f.num_planes,
                    f.width,
                    f.height,
                    f.pix_fmt,
                    f.time_base,
                    f.get_pts(),
                    f.get_timestamps(),
                )
                for f in frames
            ]
            return {"meta": meta, "video": spdl.io.convert_frames(frames)}

        self.assert_fused_matches_eager(run, data)

    def test_clone_and_slice(self) -> None:
        """Cloned packets decode twice and frames can be sliced."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> dict[str, object]:
            packets = spdl.io.demux_video(src, timestamp=(0, 1))
            gray = spdl.io.decode_packets(packets.clone(), filter_desc="format=gray")
            frames = spdl.io.decode_packets(packets)
            first = frames[0]
            return {
                "first_pts": first.pts,
                "gray": spdl.io.convert_frames(gray),
                "first": spdl.io.convert_frames(first),
                "range": spdl.io.convert_frames(frames[2:10:3]),
                "step": spdl.io.convert_frames(frames[1::4]),
                "indices": spdl.io.convert_frames(frames[[1, 4, 5]]),
                "huge_step": spdl.io.convert_frames(frames[1 :: 2**32]),
                "clone": spdl.io.convert_frames(frames.clone()),
            }

        self.assert_fused_matches_eager(run, data)

    def test_sample_decode_video(self) -> None:
        """`sample_decode_video` matches eager."""
        data = _read_sample(_GOP_VIDEO_CMD)

        def run(src: bytes) -> tuple[spdl.io.CPUBuffer, spdl.io.CPUBuffer]:
            packets = spdl.io.demux_video(src)
            sampled = spdl.io.sample_decode_video(
                packets.clone(), [0, 30, 31, 120, 199]
            )
            first = spdl.io.decode_packets(packets, num_frames=2)
            return spdl.io.convert_frames(sampled), spdl.io.convert_frames(first)

        self.assert_fused_matches_eager(run, data)

    def test_bsf(self) -> None:
        """`apply_bsf` can be fused."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> spdl.io.CPUBuffer:
            packets = spdl.io.apply_bsf(spdl.io.demux_video(src), "null")
            return spdl.io.convert_frames(spdl.io.decode_packets(packets))

        self.assert_fused_matches_eager(run, data)

    def test_default_filter_timestamp(self) -> None:
        """The default filter trims with the same timestamp as eager decoding.

        Demuxing reduces the window to a 32-bit rational, so the timestamp of
        the packets can differ from the given value.
        """
        data = _read_sample(_VIDEO_CMD)
        window = (0.011699999999999999, 1.0117)

        def run(src: bytes) -> tuple[object, object]:
            return (
                spdl.io.load_video(src, timestamp=window),
                spdl.io.load_audio(src, timestamp=window),
            )

        with mock.patch.object(
            _core, "_resolve_filter_body", wraps=_core._resolve_filter_body
        ) as resolve:
            with _eager():
                ref = run(data)
            eager_calls = list(resolve.call_args_list)
            resolve.reset_mock()
            hyp = fuse(run)(data)
            fused_calls = list(resolve.call_args_list)

        self.assertNotEqual(eager_calls[0].args[2][0], window[0])
        self.assertEqual(fused_calls, eager_calls)
        self.assert_same(hyp, ref)

    def test_get_filter_desc(self) -> None:
        """`get_filter_desc` accepts traced packets."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> tuple[object, object]:
            video = spdl.io.demux_video(src, timestamp=(0.5, 1.5))
            video_filter = spdl.io.get_filter_desc(
                video, scale_width=32, scale_height=24
            )
            audio = spdl.io.demux_audio(src, timestamp=(0.5, 1.5))
            audio_filter = spdl.io.get_filter_desc(audio, sample_rate=8000)
            return (
                spdl.io.convert_frames(
                    spdl.io.decode_packets(video, filter_desc=video_filter)
                ),
                spdl.io.convert_frames(
                    spdl.io.decode_packets(audio, filter_desc=audio_filter)
                ),
            )

        self.assert_fused_matches_eager(run, data)

    def test_per_call_arguments(self) -> None:
        """Arguments that change on every call are executed in one native call."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes, start: float) -> object:
            return spdl.io.load_video(src, timestamp=(start, start + 0.5))

        fused = fuse(run)
        for start in (0.0, 0.3, 1.1, 2.0):
            with _eager():
                ref = run(data, start)
            with mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute:
                self.assert_same(fused(data, start), ref)
            self.assertEqual(execute.call_count, 1)

    def test_composite_loaders(self) -> None:
        """`load_video` and `load_audio` can be fused."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> tuple[object, object]:
            return (
                spdl.io.load_video(src, timestamp=(0.5, 1.5)),
                spdl.io.load_audio(src, timestamp=(0.5, 1.5)),
            )

        self.assert_fused_matches_eager(run, data)


class FuseAudioTest(_FuseTestCase):
    def test_audio_metadata(self) -> None:
        """Audio packets and frames metadata match eager."""
        data = _read_sample(_AUDIO_CMD)

        def run(src: bytes) -> dict[str, object]:
            packets = spdl.io.demux_audio(src, timestamp=(0.5, 1.0))
            codec = packets.codec
            assert codec is not None
            meta: dict[str, object] = {
                "packets": (
                    packets.timestamp,
                    packets.sample_rate,
                    packets.num_channels,
                ),
                "codec": (
                    codec.name,
                    codec.sample_rate,
                    codec.num_channels,
                    codec.sample_fmt,
                    codec.channel_layout,
                    codec.time_base,
                ),
            }
            frames = spdl.io.decode_packets(packets)
            meta["frames"] = (
                frames.num_frames,
                frames.sample_rate,
                frames.num_channels,
                frames.sample_fmt,
            )
            return {"meta": meta, "audio": spdl.io.convert_frames(frames)}

        self.assert_fused_matches_eager(run, data)


class FuseLazyTest(_FuseTestCase):
    """Functions that need run-time values in Python are executed lazily."""

    def assert_native_calls(
        self,
        fn: Callable[..., Any],
        args: tuple[object, ...],
        expected: list[int],
        cache: bool = False,
    ) -> None:
        """Check the results against eager, and the number of native calls.

        The plan of a traced function is not counted, as it is executed
        without `_LazyGraph`.
        """
        with _eager():
            ref = fn(*args)
        fused = fuse(fn, cache=cache)
        counts = []
        for _ in expected:
            with mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute:
                self.assert_same(fused(*args), ref)
            counts.append(execute.call_count)
        self.assertEqual(counts, expected)

    def test_cache(self) -> None:
        """With `cache=True`, a function executed as one native call is traced."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> object:
            return spdl.io.load_video(src, timestamp=(0, 1))

        self.assert_native_calls(run, (data,), [1, 1, 1])
        self.assert_native_calls(run, (data,), [1, 0, 0], cache=True)

    def test_cache_with_array_conversion(self) -> None:
        """Converting the results to arrays does not prevent the cache."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> object:
            frames = spdl.io.decode_packets(spdl.io.demux_video(src, timestamp=(0, 1)))
            pts = frames.get_pts()
            return spdl.io.to_numpy(spdl.io.convert_frames(frames)), pts

        def permuted(src: bytes) -> object:
            buffer = spdl.io.load_video(src, timestamp=(0, 1))
            return np.transpose(spdl.io.to_numpy(buffer), (0, 3, 1, 2))

        # `to_numpy` executes the recorded operations, then the program is cached.
        self.assert_native_calls(run, (data,), [1, 0, 0], cache=True)
        # The array is used in Python, so the function cannot be cached.
        self.assert_native_calls(permuted, (data,), [1, 1, 1], cache=True)

    def test_branch_on_metadata(self) -> None:
        """Branching on metadata executes the function up to that point."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> object:
            frames = spdl.io.decode_packets(spdl.io.demux_video(src, timestamp=(0, 1)))
            if frames.num_frames == 0:
                raise ValueError("No frames")
            return spdl.io.to_numpy(spdl.io.convert_frames(frames))

        # One for `num_frames`, and one for `to_numpy`.
        self.assert_native_calls(run, (data,), [2, 2, 2])
        self.assert_native_calls(run, (data,), [2, 2, 2], cache=True)

    def test_filter_from_metadata(self) -> None:
        """Filter descriptions can be computed from metadata."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> object:
            packets = spdl.io.demux_video(src, timestamp=(0, 1))
            width, height = packets.width, packets.height
            filter_desc = spdl.io.get_video_filter_desc(
                scale_width=width // 2, scale_height=height // 2
            )
            frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
            return {"size": (width, height), "video": spdl.io.convert_frames(frames)}

        # `width` and `height` are fetched together.
        self.assert_native_calls(run, (data,), [2, 2, 2])

    def test_select_frames_from_metadata(self) -> None:
        """Frames can be selected with indices computed from metadata."""
        data = _read_sample(_GOP_VIDEO_CMD)

        def by_pts(src: bytes) -> object:
            frames = spdl.io.decode_packets(spdl.io.demux_video(src, timestamp=(1, 3)))
            pts = frames.get_pts()
            index = pts.index(pts[len(pts) // 2])
            return np.array(spdl.io.convert_frames(frames[index]))

        def sample(src: bytes) -> object:
            packets = spdl.io.demux_video(src)
            indices = list(range(0, len(packets), 40))
            return spdl.io.convert_frames(spdl.io.sample_decode_video(packets, indices))

        self.assert_native_calls(by_pts, (data,), [2, 2])
        self.assert_native_calls(sample, (data,), [2, 2])

    def test_array_from_metadata(self) -> None:
        """Metadata can be converted to arrays and tensors."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> object:
            frames = spdl.io.decode_packets(spdl.io.demux_video(src, timestamp=(0, 1)))
            pts = np.array(frames.get_pts(), dtype=np.int64)
            timestamps = torch.tensor(frames.get_timestamps())
            return pts, timestamps, spdl.io.convert_frames(frames)

        self.assert_native_calls(run, (data,), [3, 3])

    def test_values_are_released_in_native_calls(self) -> None:
        """Unused values are released in the native call, not after it.

        So that ending the call does not need to release the GIL again.
        """
        data = _read_sample(_VIDEO_CMD)
        num_values: list[int] = []

        @fuse
        def run(src: bytes) -> object:
            frames = spdl.io.decode_packets(spdl.io.demux_video(src, timestamp=(0, 1)))
            buffer = spdl.io.convert_frames(frames)
            del frames
            array = spdl.io.to_numpy(buffer)
            graph = _fuse._active_graph()
            assert isinstance(graph, _fuse._LazyGraph)
            # The demuxer is closed, the packets are decoded, the frames are
            # garbage collected, and the buffer is moved to Python.
            num_values.append(graph._state.num_values)
            return array

        run(data)
        self.assertEqual(num_values, [0])

    def test_tensor_operations(self) -> None:
        """Tensor operations on the result run in Python after conversion."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> object:
            buffer = spdl.io.load_video(src, timestamp=(0, 1))
            return spdl.io.to_torch(buffer).permute(0, 3, 1, 2).float() / 255.0

        self.assert_native_calls(run, (data,), [1, 1])


def _EXECUTE(self: object, exports: list[int]) -> list[Any]:
    return _ORIGINAL_EXECUTE(self, exports)


_ORIGINAL_EXECUTE: Callable[..., list[Any]] = _fuse._LazyGraph._execute


class FuseFallbackTest(_FuseTestCase):
    """Operations that cannot be recorded run with the actual values."""

    def test_unsupported_attribute(self) -> None:
        """Attributes that are not recorded are read from the actual value."""
        data = _read(_image_samples(1))[0]

        def run(src: bytes) -> object:
            frames = spdl.io.decode_packets(spdl.io.demux_image(src))
            return frames.metadata, spdl.io.convert_frames(frames)

        self.assert_fused_matches_eager(run, data)

    def test_streaming(self) -> None:
        """Streaming demuxing and decoding work in a fused function."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes) -> list[spdl.io.CPUBuffer]:
            with spdl.io.Demuxer(src) as demuxer:
                index = demuxer.video_stream_index
                decoder = spdl.io.Decoder(demuxer.video_codec, filter_desc=None)
                buffers = []
                for packets in demuxer.streaming_demux([index], duration=1):
                    video = packets[index]
                    assert isinstance(video, spdl.io.VideoPackets)
                    for frames in decoder.streaming_decode_packets(video):
                        buffers.append(spdl.io.convert_frames(frames))
                for frames in decoder.flush():
                    buffers.append(spdl.io.convert_frames(frames))
                return buffers

        self.assert_fused_matches_eager(run, data)

    def test_transfer(self) -> None:
        """The actual buffer is passed to the transfer to a device."""
        data = _read(_image_samples(1))[0]
        cuda = mock.NonCallableMock()
        cuda.transfer_buffer.side_effect = lambda buffer, **_: buffer

        @fuse
        def load(src: bytes) -> object:
            return spdl.io.load_image(src, device_config=mock.sentinel.device_config)

        with mock.patch.object(_core, "_libspdl_cuda", new=cuda):
            buffer = load(data)

        ((args, kwargs),) = cuda.transfer_buffer.call_args_list
        self.assertIsInstance(args[0], spdl.io.CPUBuffer)
        self.assertIs(kwargs["device_config"], mock.sentinel.device_config)
        with _eager():
            ref = spdl.io.load_image(data)
        self.assert_same(buffer, ref)

    def test_non_strict_image_batch(self) -> None:
        """`load_image_batch(strict=False)` skips images that fail to decode."""
        data = _read(_image_samples(2))

        def run(srcs: list[bytes]) -> object:
            return spdl.io.load_image_batch(srcs, width=32, height=24, strict=False)

        self.assert_fused_matches_eager(run, [data[0], b"invalid", data[1]])


_Codec = spdl.io.AudioCodec | spdl.io.VideoCodec | spdl.io.ImageCodec


def _fused_filter_graph(codec: _Codec, filter_desc: str | None) -> str | None:
    match codec:
        case spdl.io.AudioCodec():
            return _libspdl.FusedPlan.filter_graph(codec, filter_desc)
        case spdl.io.VideoCodec():
            return _libspdl.FusedPlan.filter_graph(codec, filter_desc)
        case _:
            return _libspdl.FusedPlan.filter_graph(codec, filter_desc)


def _image_codec(data: bytes) -> spdl.io.ImageCodec:
    codec = spdl.io.demux_image(data).codec
    assert codec is not None
    return codec


class FuseFilterGraphTest(unittest.TestCase):
    """Lock the filter graph of fused decoding to the one of eager decoding.

    Fused decoding attaches the codec-dependent buffer source and the sink
    in C++, while eager decoding builds the whole graph in Python.
    """

    def test_filter_graph_matches_eager(self) -> None:
        """The filter graph is the same for all media types and filters."""
        # A non-square pixel aspect ratio, an NTSC frame rate, stereo audio.
        cmd = (
            f"{FFMPEG_CLI} -hide_banner -y "
            "-f lavfi -i testsrc=size=64x48:rate=30000/1001,setsar=4/3 "
            "-f lavfi -i sine=sample_rate=22050 -ac 2 -t 2 sample.mp4"
        )
        data = _read_sample(cmd)
        image = _read(_image_samples(1))[0]
        with spdl.io.Demuxer(data) as demuxer:
            codecs: dict[str, _Codec] = {
                "audio": demuxer.audio_codec,
                "video": demuxer.video_codec,
            }
        codecs["image"] = _image_codec(image)
        custom = {"audio": "aresample=8000", "video": "scale=32:24", "image": "null"}

        for media, codec in codecs.items():
            for filter_desc in (_core._FILTER_DESC_DEFAULT, custom[media]):
                for timestamp in (None, (0.5, 1.5)):
                    with self.subTest(
                        media=media, filter_desc=filter_desc, timestamp=timestamp
                    ):
                        body = _core._resolve_filter_body(filter_desc, media, timestamp)
                        self.assertEqual(
                            _fused_filter_graph(codec, body),
                            _core._resolve_filter_graph(filter_desc, codec, timestamp),
                        )

    def test_no_filter(self) -> None:
        """A filter body of `None` disables filtering, as eager decoding does."""
        codec = _image_codec(_read(_image_samples(1))[0])

        with mock.patch.object(
            _preprocessing, "get_video_filter_desc", return_value=None
        ):
            expected = _core._resolve_filter_graph(_core._FILTER_DESC_DEFAULT, codec)
        self.assertIsNone(expected)
        self.assertIsNone(_libspdl.FusedPlan.filter_graph(codec, None))


class FusePublicAPITest(unittest.TestCase):
    def test_public_api(self) -> None:
        """`fuse` is available as `spdl.io.fuse`."""
        self.assertIs(spdl.io.fuse, fuse)
        self.assertIn("fuse", spdl.io.__all__)


class FuseErrorTest(_FuseTestCase):
    """Errors of recorded operations are raised and handled as without `fuse`."""

    def test_error_handled_in_function(self) -> None:
        """`try` in the function catches the error of a recorded operation."""
        data = _read(_image_samples(1))[0]
        calls = []

        def load(src: bytes) -> object:
            calls.append(src)
            try:
                return spdl.io.load_image(src)
            except RuntimeError:
                return None

        for cache in (False, True):
            with self.subTest(cache=cache):
                fused = fuse(load, cache=cache)
                for _ in range(3):
                    with _eager():
                        ref = load(data)
                    self.assert_same(fused(data), ref)
                calls.clear()
                self.assertIsNone(fused(b"invalid"))
                # The function runs again without recording.
                self.assertEqual(len(calls), 1 if cache else 2)

    def test_load_image_batch_logs_and_raises(self) -> None:
        """`load_image_batch` logs the images that failed, and raises."""
        data = _read(_image_samples(1))[0]

        @fuse
        def load(srcs: list[bytes]) -> object:
            return spdl.io.load_image_batch(srcs, width=32, height=24)

        with self.assertLogs("spdl.io._composite", level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "Failed to load some images"):
                load([data, b"invalid"])
        self.assertEqual(len(logs.records), 1)


class FuseSourcesTest(_FuseTestCase):
    """`sources` declares the parameters that receive sources."""

    def test_str_argument_is_part_of_the_key(self) -> None:
        """A `str` argument of other parameters makes a separate program."""
        data = _read(_image_samples(1))[0]

        def run(src: bytes, pix_fmt: str) -> object:
            return spdl.io.load_image(src, filter_desc=f"format=pix_fmts={pix_fmt}")

        fused = fuse(run, cache=True, sources=("src",))
        for pix_fmt in ("rgb24", "gray", "rgb24", "gray", "rgb24", "gray"):
            with _eager():
                ref = run(data, pix_fmt)
            self.assert_same(fused(data, pix_fmt), ref)
        # Without `sources`, the `str` is classified as a source, and the
        # function is not cached. With `sources`, each value is cached.
        with mock.patch.object(
            _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
        ) as execute:
            fused(data, "gray")
        self.assertEqual(execute.call_count, 0)

    def test_sources_in_containers(self) -> None:
        """Sources nested in lists are sources, other values are constants."""
        images = _read(_image_samples(2))

        def run(srcs: list[bytes], sizes: list[str]) -> object:
            return [
                spdl.io.load_image(s, filter_desc=f"scale={z}")
                for s, z in zip(srcs, sizes, strict=True)
            ]

        fused = fuse(run, cache=True, sources=("srcs",))
        for _ in range(3):
            for sizes in (["16:12", "8:6"], ["8:6", "16:12"]):
                with _eager():
                    ref = run(images, sizes)
                self.assert_same(fused(images, sizes), ref)

    def test_invalid_sources(self) -> None:
        """`sources` must name parameters, and requires `cache=True`."""

        def run(src: bytes) -> object:
            return spdl.io.load_image(src)

        with self.assertRaisesRegex(ValueError, "must be parameters"):
            fuse(run, cache=True, sources=("data",))
        with self.assertRaisesRegex(ValueError, "only with `cache=True`"):
            fuse(run, sources=("src",))


class FuseCacheSourceTest(_FuseTestCase):
    """Arguments classified as sources by mistake are not cached."""

    def test_buffer_argument_that_is_not_a_source(self) -> None:
        """A NumPy array used as a value is not baked into the program."""
        data = _read_sample(_VIDEO_CMD)

        def run(src: bytes, window: np.ndarray) -> object:
            return spdl.io.load_video(
                src, timestamp=(float(window[0]), float(window[1]))
            )

        fused = fuse(run, cache=True)
        for _ in range(2):
            for window in ((0.0, 0.5), (1.0, 2.0), (0.3, 1.1)):
                arg = np.array(window)
                with _eager():
                    ref = run(data, arg)
                self.assert_same(fused(data, arg), ref)

    def test_str_argument_that_is_not_a_source(self) -> None:
        """A `str` formatted into a filter is not baked into the program."""
        data = _read(_image_samples(1))[0]

        def run(src: bytes, pix_fmt: str) -> object:
            return spdl.io.load_image(src, filter_desc=f"format=pix_fmts={pix_fmt}")

        fused = fuse(run, cache=True)
        for _ in range(3):
            for pix_fmt in ("rgb24", "gray"):
                with _eager():
                    ref = run(data, pix_fmt)
                self.assert_same(fused(data, pix_fmt), ref)

    def test_source_types(self) -> None:
        """`bytes`, `bytearray`, `memoryview`, `str` and `Path` are sources."""
        samples = _image_samples(1)
        data = _read(samples)[0]
        path = samples[0].path
        with _eager():
            ref = spdl.io.load_image(data)

        def load(src: Any) -> object:
            # `load_image` accepts any source type that `Demuxer` accepts.
            return spdl.io.load_image(src)

        for cache in (False, True):
            fused = fuse(load, cache=cache)
            for src in (data, bytearray(data), memoryview(data), path, Path(path)):
                with self.subTest(cache=cache, src=type(src)):
                    for _ in range(3):
                        self.assert_same(fused(src), ref)


class FuseLoaderTest(_FuseTestCase):
    """The loading functions of `spdl.io` are fused."""

    def test_direct_call_is_one_native_call(self) -> None:
        """A direct call matches the unfused function, in one native call."""
        data = _read_sample(_VIDEO_CMD)
        images = _read(_image_samples(3))
        cases: list[tuple[Callable[..., Any], tuple[object, ...], dict[str, Any]]] = [
            (spdl.io.load_audio, (data,), {}),
            (spdl.io.load_audio, (data,), {"timestamp": (0.5, 1.5)}),
            (spdl.io.load_video, (data,), {"timestamp": (0.5, 1.5)}),
            (spdl.io.load_video, (data,), {"filter_desc": "scale=32:24,format=gray"}),
            (spdl.io.load_image, (images[0],), {}),
            (spdl.io.load_image, (images[0],), {"filter_desc": None}),
            (spdl.io.load_image_batch, (images,), {"width": 32, "height": 24}),
        ]
        for loader, args, kwargs in cases:
            with self.subTest(loader=loader.__name__, kwargs=kwargs):
                ref = inspect.unwrap(loader)(*args, **kwargs)
                with mock.patch.object(
                    _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
                ) as execute:
                    hyp = loader(*args, **kwargs)
                self.assertEqual(execute.call_count, 1)
                self.assert_same(hyp, ref)

    def test_errors_are_raised_as_without_fuse(self) -> None:
        """A failed decoding raises the same error as the unfused function."""
        for loader in (spdl.io.load_audio, spdl.io.load_video, spdl.io.load_image):
            with self.subTest(loader=loader.__name__):
                with self.assertRaises(RuntimeError) as ref:
                    inspect.unwrap(loader)(b"invalid")
                with self.assertRaises(RuntimeError) as hyp:
                    loader(b"invalid")
                self.assertEqual(str(hyp.exception), str(ref.exception))

    def test_image_batch_failures(self) -> None:
        """`load_image_batch` logs each failed image once, as without `fuse`."""
        good = _read(_image_samples(2))
        srcs = [good[0], b"invalid", good[1]]

        with self.assertLogs("spdl.io._composite", level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "Failed to load some images"):
                spdl.io.load_image_batch(srcs, width=32, height=24)
        self.assertEqual(len(logs.records), 1)

        with self.assertLogs("spdl.io._composite", level="ERROR") as logs:
            hyp = spdl.io.load_image_batch(srcs, width=32, height=24, strict=False)
        self.assertEqual(len(logs.records), 1)
        ref = inspect.unwrap(spdl.io.load_image_batch)(
            [good[0], good[1]], width=32, height=24
        )
        self.assert_same(hyp, ref)

    def test_runtime_switch(self) -> None:
        """The switch receives the name of the loader, and disables it."""
        image = _read(_image_samples(1))[0]
        names: list[str] = []

        def disabled(name: str) -> bool:
            names.append(name)
            return True

        # The switch is read when the loader is decorated, so decorate again.
        with mock.patch.object(_fuse, "_is_fusion_disabled", side_effect=disabled):
            loader = fuse(
                inspect.unwrap(spdl.io.load_image), cache=True, sources=("src",)
            )
        with mock.patch.object(
            _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
        ) as execute:
            hyp = loader(image)
        self.assertEqual(names, ["spdl.io._composite.load_image"])
        self.assertEqual(execute.call_count, 0)
        self.assert_same(hyp, inspect.unwrap(spdl.io.load_image)(image))

    def test_cached(self) -> None:
        """Repeated calls run the cached program, for each set of arguments."""
        data = _read_sample(_VIDEO_CMD)
        images = _read(_image_samples(3))
        cases: list[tuple[Callable[..., Any], tuple[object, ...], dict[str, Any]]] = [
            (spdl.io.load_audio, (data,), {"timestamp": (0.5, 1.5)}),
            (spdl.io.load_video, (data, (0.5, 1.5)), {}),
            (spdl.io.load_image, (images[0],), {"filter_desc": "format=gray"}),
            (spdl.io.load_image, (images[0],), {"filter_desc": "format=rgb24"}),
            (spdl.io.load_image_batch, (images,), {"width": 32, "height": 24}),
            (spdl.io.load_image_batch, (images[:2],), {"width": 16, "height": 12}),
        ]
        for loader, args, kwargs in cases:
            with self.subTest(loader=loader.__name__, args=len(args), kwargs=kwargs):
                with _eager():
                    ref = loader(*args, **kwargs)
                counts = []
                for _ in range(4):
                    with mock.patch.object(
                        _fuse._LazyGraph,
                        "_execute",
                        autospec=True,
                        side_effect=_EXECUTE,
                    ) as execute:
                        self.assert_same(loader(*args, **kwargs), ref)
                    counts.append(execute.call_count)
                # Recorded, traced, then cached.
                self.assertEqual(counts, [1, 0, 0, 0])

    def test_cached_with_different_sources(self) -> None:
        """A cached program is used for different sources."""
        images = _read(_image_samples(3))
        for image in images:
            spdl.io.load_image(image, filter_desc="scale=16:12")
        for image in images:
            with _eager():
                ref = spdl.io.load_image(image, filter_desc="scale=16:12")
            with mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute:
                self.assert_same(
                    spdl.io.load_image(image, filter_desc="scale=16:12"), ref
                )
            self.assertEqual(execute.call_count, 0)

    def test_cache_from_multiple_threads(self) -> None:
        """Calls from multiple threads, which evict programs, are correct."""
        images = _read(_image_samples(2))
        filters = [f"scale={8 * (i + 1)}:{6 * (i + 1)}" for i in range(6)]
        with _eager():
            refs = {
                (i, f): spdl.io.load_image(images[i], filter_desc=f)
                for i in range(2)
                for f in filters
            }
        work = [(i, f) for _ in range(10) for i in range(2) for f in filters]
        with (
            mock.patch.object(_fuse, "_CACHE_SIZE", 2),
            ThreadPoolExecutor(8) as pool,
        ):
            results = list(
                pool.map(
                    lambda w: spdl.io.load_image(images[w[0]], filter_desc=w[1]), work
                )
            )
        for w, result in zip(work, results, strict=True):
            self.assert_same(result, refs[w])

    def test_reference_is_eager(self) -> None:
        """`_eager` disables fusion, so that references do not record."""
        image = _read(_image_samples(1))[0]
        with (
            _eager(),
            mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute,
        ):
            spdl.io.load_image(image)
        self.assertEqual(execute.call_count, 0)

    def test_signature_and_pickle(self) -> None:
        """The loaders keep their signatures, docs and pickling by reference."""
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                original = inspect.unwrap(loader)
                self.assertIsNot(loader, original)
                self.assertEqual(inspect.signature(loader), inspect.signature(original))
                self.assertEqual(loader.__doc__, original.__doc__)
                self.assertEqual(loader.__module__, "spdl.io._composite")
                self.assertIs(pickle.loads(pickle.dumps(loader)), loader)


class FuseTracingErrorTest(unittest.TestCase):
    def test_packets_used_after_decode(self) -> None:
        """Using packets after decoding consumed them is rejected at trace time."""

        @fuse
        def load(src: bytes) -> list[spdl.io.CPUBuffer]:
            packets = spdl.io.demux_image(src)
            return [
                spdl.io.convert_frames(spdl.io.decode_packets(packets))
                for _ in range(2)
            ]

        with self.assertRaisesRegex(ValueError, "used after decoding"):
            load(b"dummy")

    def test_packets_used_after_sample_decode(self) -> None:
        """`sample_decode_video` consumes packets as it does eagerly."""

        @fuse
        def load(src: bytes) -> object:
            packets = spdl.io.demux_video(src)
            spdl.io.sample_decode_video(packets, [0])
            return spdl.io.convert_frames(spdl.io.decode_packets(packets))

        with self.assertRaisesRegex(ValueError, "used after sample decoding"):
            load(b"dummy")

    def test_python_object_output(self) -> None:
        """Python objects and containers in the result are preserved."""
        data = _read(_image_samples(1))[0]
        Pair = collections.namedtuple("Pair", ["image", "label"])

        @dataclasses.dataclass
        class Sample:
            image: object

        @fuse
        def load(src: bytes) -> tuple[Pair, Sample]:
            return (
                Pair(spdl.io.load_image(src), "x"),
                Sample(spdl.io.load_image(src)),
            )

        with _eager():
            ref = spdl.io.to_numpy(spdl.io.load_image(data))
        for _ in range(2):
            pair, sample = load(data)
            self.assertIsInstance(pair, Pair)
            self.assertIsInstance(pair.image, spdl.io.CPUBuffer)
            self.assertEqual(pair.label, "x")
            np.testing.assert_array_equal(
                spdl.io.to_numpy(pair.image), ref, strict=True
            )
            np.testing.assert_array_equal(np.asarray(sample.image), ref, strict=True)


class FuseRuntimeSwitchTest(unittest.TestCase):
    def test_disabled_when_decorated(self) -> None:
        """When fusion is disabled at decoration, the function runs eagerly."""
        data = _read(_image_samples(1))[0]

        def load(src: bytes) -> spdl.io.CPUBuffer:
            packets = spdl.io.demux_image(src)
            self.assertIsInstance(packets, spdl.io.ImagePackets)
            return spdl.io.convert_frames(spdl.io.decode_packets(packets))

        name = (
            f"{__name__}.FuseRuntimeSwitchTest.test_disabled_when_decorated"
            ".<locals>.load"
        )
        for cache in (False, True):
            with self.subTest(cache=cache):
                with mock.patch.object(
                    _fuse, "_is_fusion_disabled", return_value=True
                ) as disabled:
                    fused = fuse(load, cache=cache)
                self.assertEqual(disabled.call_args_list, [mock.call(name)])
                results = [fused(data) for _ in range(3)]
                for result in results:
                    self.assertIsInstance(result, spdl.io.CPUBuffer)
                    np.testing.assert_array_equal(
                        spdl.io.to_numpy(result),
                        spdl.io.to_numpy(load(data)),
                        strict=True,
                    )

    def test_read_once(self) -> None:
        """The switch is not read when the function is called."""
        data = _read(_image_samples(1))[0]
        seen: list[str] = []

        def load(src: bytes) -> object:
            seen.append(type(spdl.io.demux_image(src)).__name__)
            return None

        for cache in (False, True):
            with self.subTest(cache=cache):
                seen.clear()
                fused = fuse(load, cache=cache)
                with mock.patch.object(
                    _fuse,
                    "_is_fusion_disabled",
                    side_effect=AssertionError("read on call"),
                ):
                    fused(data)
                self.assertEqual(seen, ["_Symbol"])


class FuseFreeThreadingTest(unittest.TestCase):
    def test_noop_without_gil(self) -> None:
        """Without the GIL, the function runs eagerly."""
        data = _read(_image_samples(1))[0]

        def load(src: bytes) -> spdl.io.CPUBuffer:
            packets = spdl.io.demux_image(src)
            self.assertIsInstance(packets, spdl.io.ImagePackets)
            return spdl.io.convert_frames(spdl.io.decode_packets(packets))

        for cache in (False, True):
            with self.subTest(cache=cache):
                fused = fuse(load, cache=cache)
                with mock.patch.object(_fuse, "_gil_enabled", return_value=False):
                    results = [fused(data) for _ in range(3)]
                for result in results:
                    self.assertIsInstance(result, spdl.io.CPUBuffer)
                    np.testing.assert_array_equal(
                        spdl.io.to_numpy(result),
                        spdl.io.to_numpy(load(data)),
                        strict=True,
                    )
