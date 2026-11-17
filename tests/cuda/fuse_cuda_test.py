# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import gc
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import spdl.io
import spdl.io.utils
import torch
from spdl.io import _core, _fuse
from spdl.io._fuse import fuse

from ..fixture import FFMPEG_CLI, get_sample, get_samples, SrcInfo

_ORIGINAL_EXECUTE = _fuse._LazyGraph._execute
_ORIGINAL_PROGRAM_RUN = _fuse._Program.run


def _EXECUTE(graph: Any, exports: list[int]) -> list[Any]:
    return _ORIGINAL_EXECUTE(graph, exports)


def setUpModule() -> None:
    for name, value in (("_gil_enabled", True), ("_is_fusion_disabled", False)):
        patcher = mock.patch.object(_fuse, name, return_value=value)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


@contextlib.contextmanager
def _eager() -> Iterator[None]:
    # `fuse` does nothing without the GIL, which is checked on every call.
    # (The switch is read only when a function is decorated.)
    with mock.patch.object(_fuse, "_gil_enabled", return_value=False):
        yield


def _load_image(src: bytes, device_config: spdl.io.CUDAConfig) -> Any:
    return spdl.io.load_image(src, device_config=device_config)


def _transfer_image(src: bytes, device_config: spdl.io.CUDAConfig) -> Any:
    frames = spdl.io.decode_packets(spdl.io.demux_image(src))
    return spdl.io.transfer_buffer(
        spdl.io.convert_frames(frames), device_config=device_config
    )


def _decode_video(src: bytes, device_config: spdl.io.CUDAConfig) -> Any:
    return spdl.io.decode_packets_nvdec(
        spdl.io.demux_video(src), device_config=device_config
    )


def _nv12_cpu(src: bytes) -> Any:
    frames = spdl.io.decode_packets(spdl.io.demux_video(src), filter_desc="format=nv12")
    return spdl.io.convert_frames(frames[0])


def _convert_nv12(
    src: bytes,
    device_config: spdl.io.CUDAConfig,
    *,
    bgr: bool = False,
    coeff: int = 1,
    sync: bool = False,
) -> Any:
    cpu = _nv12_cpu(src)
    cuda = spdl.io.transfer_buffer(cpu, device_config=device_config)
    convert = spdl.io.nv12_to_bgr if bgr else spdl.io.nv12_to_rgb
    return convert(cuda, device_config=device_config, coeff=coeff, sync=sync)


class _FuseCudaTest(unittest.TestCase):
    images: list[SrcInfo]
    video: SrcInfo
    image_data: list[bytes]
    video_data: bytes

    @classmethod
    def setUpClass(cls) -> None:
        if not spdl.io.utils.built_with_cuda() or not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA support and a CUDA device are required")
        cls.images = sorted(
            get_samples(
                f"{FFMPEG_CLI} -hide_banner -y -f lavfi "
                "-i testsrc=size=64x48:rate=4 -frames:v 3 sample_%03d.png"
            ),
            key=lambda sample: sample.path,
        )
        cls.addClassCleanup(cls.images[0]._tmp_dir.cleanup)
        cls.video = get_sample(
            f"{FFMPEG_CLI} -hide_banner -y -f lavfi "
            "-i testsrc=size=64x48:rate=4,format=yuv420p "
            "-c:v libx264 -frames:v 8 -g 4 -bf 0 sample.mp4"
        )
        cls.addClassCleanup(cls.video._tmp_dir.cleanup)
        cls.image_data = [Path(sample.path).read_bytes() for sample in cls.images]
        cls.video_data = Path(cls.video.path).read_bytes()

    def setUp(self) -> None:
        self.device_config = spdl.io.cuda_config(
            device_index=0, stream=torch.cuda.current_stream(0).cuda_stream
        )
        for loader in (
            spdl.io.load_image,
            spdl.io.load_video,
            spdl.io.load_image_batch,
        ):
            # The loading functions are cached once they are fused themselves.
            if clear := vars(loader).get("_clear_cache"):
                clear()
        self.addCleanup(_core._del_cached_decoder)

    def _cpu(self, result: Any) -> torch.Tensor:
        tensor = (
            result if isinstance(result, torch.Tensor) else spdl.io.to_torch(result)
        )
        self.assertTrue(tensor.is_cuda)
        self.assertEqual(tensor.device.index, 0)
        return tensor.cpu()

    def _assert_modes(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        with _eager():
            ref = self._cpu(fn(*args, **kwargs))
        for cache in (False, True):
            with self.subTest(cache=cache):
                self._assert_fused(fn, ref, args, kwargs, cache)

    def _assert_fused(
        self,
        fn: Callable[..., Any],
        ref: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cache: bool,
    ) -> None:
        fused = fuse(fn, cache=cache)
        counts = []
        with (
            mock.patch.object(_core, "_libspdl_cuda", new=self._cuda_factories()),
            mock.patch.object(
                _fuse._Program,
                "run",
                autospec=True,
                side_effect=_ORIGINAL_PROGRAM_RUN,
            ) as program_run,
        ):
            for _ in range(3):
                with mock.patch.object(
                    _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
                ) as execute:
                    hyp = self._cpu(fused(*args, **kwargs))
                torch.testing.assert_close(hyp, ref, rtol=0, atol=0)
                counts.append(execute.call_count)
        self.assertEqual(counts, [1, 0, 0] if cache else [1, 1, 1])
        self.assertEqual(program_run.call_count, 2 if cache else 0)

    def _cuda_factories(self) -> Any:
        names = ("_fused_transfer_buffer", "_fused_decode_nvdec", "_fused_nv12_convert")
        cuda = mock.NonCallableMock(spec=names)
        for name in names:
            setattr(cuda, name, getattr(_core._libspdl_cuda, name, None))
        return cuda

    def _require_nvdec(self) -> None:
        if not spdl.io.utils.built_with_nvcodec():
            self.skipTest("SPDL is not compiled with NVCODEC support")

    def _require_nv12(self) -> None:
        try:
            with _eager():
                buffer = _nv12_cpu(self.video_data)
        except RuntimeError as error:
            if "Unsupported pixel format: nv12" in str(error):
                self.skipTest("This FFmpeg variant cannot convert NV12 frames")
            raise
        self.assertEqual(spdl.io.to_torch(buffer).shape, (1, 72, 64))


class FuseCudaLoaderTest(_FuseCudaTest):
    def _loader_cases(self) -> list[tuple[Callable[..., Any], Any, dict[str, Any]]]:
        return [
            (spdl.io.load_image, self.image_data[0], {}),
            (spdl.io.load_image, self.image_data[1], {"filter_desc": "scale=32:24"}),
            (spdl.io.load_video, self.video_data, {"timestamp": (0.5, 1.5)}),
            (
                spdl.io.load_image_batch,
                self.image_data,
                {"width": 32, "height": 24},
            ),
        ]

    def test_composite_loaders(self) -> None:
        """Image, video and image-batch CUDA loaders match eager in every mode."""
        for loader, src, kwargs in self._loader_cases():
            with self.subTest(loader=loader.__name__, kwargs=kwargs):

                def load(
                    source: Any,
                    device_config: spdl.io.CUDAConfig,
                    loader: Callable[..., Any] = loader,
                    kwargs: dict[str, Any] = kwargs,
                ) -> Any:
                    return loader(source, device_config=device_config, **kwargs)

                self._assert_modes(load, src, self.device_config)

    def test_explicit_transfer(self) -> None:
        """Demux, decode, convert and explicit CUDA transfer remain one native call."""
        self._assert_modes(_transfer_image, self.image_data[0], self.device_config)

    def test_to_torch(self) -> None:
        """CUDA tensor conversion preserves pixels and allows native program caching."""

        def load(src: bytes, device_config: spdl.io.CUDAConfig) -> torch.Tensor:
            return spdl.io.to_torch(_transfer_image(src, device_config))

        self._assert_modes(load, self.image_data[0], self.device_config)

    def test_device_config_identity_and_new_sources(self) -> None:
        """CUDA config identities select distinct programs that accept new sources."""
        configs = [self.device_config, spdl.io.cuda_config(device_index=0, stream=0)]
        self.assertIsNot(configs[0], configs[1])
        for config in configs:
            self.assertFalse(_fuse._is_source(config))
        fused = fuse(_load_image, cache=True)
        programs = []
        for config in configs:
            with self.subTest(config=config):
                program = self._warm_config(fused, config)
                programs.append(program)
        self.assertIsNot(programs[0], programs[1])
        with mock.patch.object(
            _fuse, "_try_trace", side_effect=AssertionError("retrace")
        ):
            for config, program in zip(configs, programs, strict=True):
                self._assert_cached_source(fused, config, program)

    def _warm_config(
        self, fused: Callable[..., Any], config: spdl.io.CUDAConfig
    ) -> Any:
        with mock.patch.object(
            _fuse._Program, "run", autospec=True, side_effect=_ORIGINAL_PROGRAM_RUN
        ) as run:
            for src in self.image_data[:2]:
                with _eager():
                    ref = self._cpu(_load_image(src, config))
                torch.testing.assert_close(
                    self._cpu(fused(src, config)), ref, rtol=0, atol=0
                )
        self.assertEqual(run.call_count, 1)
        return run.call_args.args[0]

    def _assert_cached_source(
        self, fused: Callable[..., Any], config: spdl.io.CUDAConfig, program: Any
    ) -> None:
        src = self.image_data[2]
        with _eager():
            ref = self._cpu(_load_image(src, config))
        with (
            mock.patch.object(
                _fuse._Program, "run", autospec=True, side_effect=_ORIGINAL_PROGRAM_RUN
            ) as run,
            mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute,
        ):
            torch.testing.assert_close(
                self._cpu(fused(src, config)), ref, rtol=0, atol=0
            )
        self.assertEqual(execute.call_count, 0)
        run.assert_called_once()
        self.assertIs(run.call_args.args[0], program)

    def test_metadata_materializes_cuda_buffer(self) -> None:
        """Packet and CUDA array metadata materialize without losing the GPU output."""

        def load(src: bytes, device_config: spdl.io.CUDAConfig) -> tuple[Any, Any]:
            packets = spdl.io.demux_image(src)
            width, height = packets.width, packets.height
            cuda = spdl.io.transfer_buffer(
                spdl.io.convert_frames(spdl.io.decode_packets(packets)),
                device_config=device_config,
            )
            meta = (
                int(width),
                int(height),
                cuda.device_index,
                cuda.__cuda_array_interface__["shape"],
            )
            return meta, spdl.io.to_torch(cuda)

        with _eager():
            ref_meta, ref = load(self.image_data[0], self.device_config)
            ref = self._cpu(ref)
        for cache in (False, True):
            fused = fuse(load, cache=cache)
            for _ in range(3):
                with mock.patch.object(
                    _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
                ) as execute:
                    meta, tensor = fused(self.image_data[0], self.device_config)
                self.assertEqual(meta, ref_meta)
                torch.testing.assert_close(self._cpu(tensor), ref, rtol=0, atol=0)
                self.assertEqual(execute.call_count, 2)

    def test_concrete_transfer_fallback(self) -> None:
        """Preexisting CPU buffers transfer eagerly with zero executor calls."""
        with _eager():
            cpu = spdl.io.load_image(self.image_data[0])
            ref = self._cpu(
                spdl.io.transfer_buffer(cpu, device_config=self.device_config)
            )

        def transfer(buffer: Any, device_config: spdl.io.CUDAConfig) -> Any:
            return spdl.io.transfer_buffer(buffer, device_config=device_config)

        for cache in (False, True):
            fused = fuse(transfer, cache=cache)
            with mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute:
                for _ in range(3):
                    with _eager():
                        cpu = spdl.io.load_image(self.image_data[0])
                    torch.testing.assert_close(
                        self._cpu(fused(cpu, self.device_config)), ref, rtol=0, atol=0
                    )
            self.assertEqual(execute.call_count, 0)


class FuseCudaVideoTest(_FuseCudaTest):
    def test_nvdec_automatic_and_explicit_annexb(self) -> None:
        """NVDEC matches eager RGB/BGR with automatic or explicit Annex B filtering."""
        self._require_nvdec()
        for bsf in (False, True):
            for pix_fmt in ("rgb", "bgr"):
                self._assert_nvdec_case(bsf, pix_fmt)

    def test_out_of_range_crop_raises(self) -> None:
        """A crop that does not fit the decoder's crop type raises, eager or fused."""
        self._require_nvdec()

        def load(src: bytes, device_config: spdl.io.CUDAConfig) -> Any:
            packets = spdl.io.demux_video(src, timestamp=(0.5, 1.5))
            return spdl.io.decode_packets_nvdec(
                packets, device_config=device_config, crop_left=1 << 16
            )

        for fused in (False, True):
            fn = spdl.io.fuse(load) if fused else load
            with self.subTest(fused=fused), self.assertRaises(ValueError):
                fn(self.video_data, self.device_config)

    def _assert_nvdec_case(self, bsf: bool, pix_fmt: str) -> None:
        with self.subTest(explicit_bsf=bsf, pix_fmt=pix_fmt):

            def load(src: bytes, device_config: spdl.io.CUDAConfig) -> Any:
                packets = spdl.io.demux_video(src, timestamp=(0.5, 1.5))
                if bsf:
                    filtered = spdl.io.apply_bsf(packets, "h264_mp4toannexb")
                    assert filtered is not None
                    packets = filtered
                return spdl.io.decode_packets_nvdec(
                    packets,
                    device_config=device_config,
                    pix_fmt=pix_fmt,
                    crop_left=4,
                    crop_top=2,
                    crop_right=4,
                    crop_bottom=2,
                    scale_width=32,
                    scale_height=24,
                )

            with _eager():
                ref = self._cpu(load(self.video_data, self.device_config))
            self.assertEqual(ref.shape, (4, 3, 24, 32))
            self._assert_modes(load, self.video_data, self.device_config)

    def test_nv12_conversions(self) -> None:
        """Transferred NV12 converts to planar RGB and BGR with eager pixel parity."""
        self._require_nv12()
        for bgr, coeff, sync in ((False, 1, False), (True, 6, True)):
            with self.subTest(bgr=bgr, coeff=coeff, sync=sync):
                self._assert_modes(
                    _convert_nv12,
                    self.video_data,
                    self.device_config,
                    bgr=bgr,
                    coeff=coeff,
                    sync=sync,
                )

    def test_concrete_nvdec_fallback(self) -> None:
        """Preexisting video packets decode eagerly without executing a fusion graph."""
        self._require_nvdec()

        def decode(packets: Any, device_config: spdl.io.CUDAConfig) -> Any:
            return spdl.io.decode_packets_nvdec(packets, device_config=device_config)

        with _eager():
            ref = self._cpu(_decode_video(self.video_data, self.device_config))
        for cache in (False, True):
            fused = fuse(decode, cache=cache)
            for _ in range(3):
                with _eager():
                    packets = spdl.io.demux_video(self.video_data)
                with mock.patch.object(
                    _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
                ) as execute:
                    hyp = self._cpu(fused(packets, self.device_config))
                torch.testing.assert_close(hyp, ref, rtol=0, atol=0)
                self.assertEqual(execute.call_count, 0)

    def test_concrete_nv12_fallback(self) -> None:
        """Preexisting CUDA NV12 buffers convert eagerly with zero executor calls."""
        self._require_nv12()
        with _eager():
            cpu = _nv12_cpu(self.video_data)
            cuda = spdl.io.transfer_buffer(cpu, device_config=self.device_config)
        for convert in (spdl.io.nv12_to_rgb, spdl.io.nv12_to_bgr):
            with _eager():
                ref = self._cpu(convert(cuda, device_config=self.device_config))
            self._assert_concrete_conversion(convert, cuda, ref)

    def _assert_concrete_conversion(
        self, convert: Callable[..., Any], cuda: Any, ref: torch.Tensor
    ) -> None:
        for cache in (False, True):
            fused = fuse(convert, cache=cache)
            with mock.patch.object(
                _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
            ) as execute:
                for _ in range(3):
                    hyp = self._cpu(fused(cuda, device_config=self.device_config))
                    torch.testing.assert_close(hyp, ref, rtol=0, atol=0)
            self.assertEqual(execute.call_count, 0)


class FuseCudaAllocatorTest(_FuseCudaTest):
    def _allocator(
        self, allocated: list[int], deleted: list[int]
    ) -> tuple[Callable[[int, int, int], int], Callable[[int], None]]:
        def allocate(size: int, device: int, stream: int) -> int:
            self.assertGreater(size, 0)
            self.assertEqual(device, 0)
            self.assertEqual(stream, torch.cuda.current_stream(0).cuda_stream)
            ptr = torch.cuda.caching_allocator_alloc(size, device, stream)
            allocated.append(ptr)
            return ptr

        def delete(ptr: int) -> None:
            deleted.append(ptr)
            torch.cuda.caching_allocator_delete(ptr)

        return allocate, delete

    def _assert_allocator(self, fn: Callable[..., Any], src: bytes) -> None:
        allocated: list[int] = []
        deleted: list[int] = []
        config = spdl.io.cuda_config(
            device_index=0,
            stream=torch.cuda.current_stream(0).cuda_stream,
            allocator=self._allocator(allocated, deleted),
        )
        self._assert_modes(fn, src, config)
        gc.collect()
        self.assertGreater(len(allocated), 0)
        self.assertCountEqual(deleted, allocated)

    def test_transfer_allocator_callbacks(self) -> None:
        """Fused transfer safely reenters Python for torch allocation and deletion."""
        self._assert_allocator(_transfer_image, self.image_data[0])

    def test_nvdec_allocator_callbacks(self) -> None:
        """Fused NVDEC safely invokes Python allocation and deletion callbacks."""
        self._require_nvdec()
        self._assert_allocator(_decode_video, self.video_data)

    def test_nv12_allocator_callbacks(self) -> None:
        """Fused color conversion safely invokes Python torch allocator callbacks."""
        self._require_nv12()
        self._assert_allocator(_convert_nv12, self.video_data)

    def test_nvdec_allocator_reentrancy(self) -> None:
        """A cached NVDEC allocator can run another cached decode on the same thread."""
        self._require_nvdec()
        inner = fuse(_decode_video, cache=True)
        for _ in range(2):
            self._cpu(inner(self.video_data, self.device_config))
        with _eager():
            ref = self._cpu(_decode_video(self.video_data, self.device_config))
        reenter = False
        nested = []

        def allocate(size: int, device: int, stream: int) -> int:
            nonlocal reenter
            if reenter:
                reenter = False
                nested.append(self._cpu(inner(self.video_data, self.device_config)))
            return torch.cuda.caching_allocator_alloc(size, device, stream)

        config = spdl.io.cuda_config(
            device_index=0,
            stream=torch.cuda.current_stream(0).cuda_stream,
            allocator=(allocate, torch.cuda.caching_allocator_delete),
        )
        outer = fuse(_decode_video, cache=True)
        for _ in range(2):
            self._cpu(outer(self.video_data, config))
        reenter = True
        with (
            mock.patch.object(
                _fuse._Program, "run", autospec=True, side_effect=_ORIGINAL_PROGRAM_RUN
            ) as run,
            mock.patch.object(
                _core,
                "nvdec_decoder",
                side_effect=AssertionError("unexpected eager retry"),
            ),
        ):
            hyp = self._cpu(outer(self.video_data, config))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(len(nested), 1)
        torch.testing.assert_close(hyp, ref, rtol=0, atol=0)
        torch.testing.assert_close(nested[0], ref, rtol=0, atol=0)

    def test_native_failure_retries_eagerly(self) -> None:
        """A native CUDA allocation failure retries the function eagerly."""
        for cache in (False, True):
            with self.subTest(cache=cache):
                self._assert_native_retry(cache)

    def _assert_native_retry(self, cache: bool) -> None:
        fail = False
        failures = []
        calls = []

        def allocate(size: int, device: int, stream: int) -> int:
            nonlocal fail
            if fail:
                fail = False
                failures.append("allocation failed")
                raise RuntimeError("intentional CUDA allocation failure")
            return torch.cuda.caching_allocator_alloc(size, device, stream)

        config = spdl.io.cuda_config(
            device_index=0,
            stream=torch.cuda.current_stream(0).cuda_stream,
            allocator=(allocate, torch.cuda.caching_allocator_delete),
        )

        def load(src: bytes, device_config: spdl.io.CUDAConfig) -> Any:
            calls.append(_fuse._active_graph() is None)
            return _transfer_image(src, device_config)

        with _eager():
            ref = self._cpu(load(self.image_data[0], config))
        fused = fuse(load, cache=cache)
        if cache:
            for _ in range(2):
                self._cpu(fused(self.image_data[0], config))
        calls.clear()
        fail = True
        with mock.patch.object(
            _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
        ) as execute:
            hyp = self._cpu(fused(self.image_data[0], config))
        torch.testing.assert_close(hyp, ref, rtol=0, atol=0)
        self.assertEqual(failures, ["allocation failed"])
        self.assertEqual(calls, [True] if cache else [False, True])
        self.assertEqual(execute.call_count, 0 if cache else 1)


class FuseCudaRejectionTest(_FuseCudaTest):
    def test_cpu_only_operations_reject_cuda_symbols(self) -> None:
        """CPU-only operations reject CUDA symbols without native execution."""
        for operation in ("convert_frames", "transfer_buffer"):
            with self.subTest(operation=operation):
                self._assert_cpu_rejection(operation)

    def _assert_cpu_rejection(self, operation: str) -> None:
        def load(src: bytes) -> Any:
            cuda: Any = spdl.io.load_image(src, device_config=self.device_config)
            if operation == "convert_frames":
                return spdl.io.convert_frames(cuda)
            return spdl.io.transfer_buffer(cuda, device_config=self.device_config)

        with mock.patch.object(
            _fuse._LazyGraph, "_execute", autospec=True, side_effect=_EXECUTE
        ) as execute:
            with self.assertRaisesRegex(
                (TypeError, RuntimeError), "(?i)expected frames|expected buffer"
            ):
                fuse(load)(b"not decoded")
        self.assertEqual(execute.call_count, 0)
