# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import threading
import unittest
from unittest.mock import MagicMock, patch

import spdl.io
import spdl.io.utils
import torch

from ..fixture import FFMPEG_CLI, get_sample

if not spdl.io.utils.built_with_nvcodec():
    raise unittest.SkipTest(  # pyre-ignore: [29]
        "SPDL is not compiled with NVCODEC support"
    )


DEFAULT_CUDA = 0


def _decode_video(src, timestamp=None, allocator=None, **decode_options):
    device_config = spdl.io.cuda_config(
        device_index=DEFAULT_CUDA,
        allocator=allocator,
    )
    packets = spdl.io.demux_video(src, timestamp=timestamp)
    buffer = spdl.io.decode_packets_nvdec(
        packets, device_config=device_config, **decode_options
    )
    return spdl.io.to_torch(buffer)


def _get_dummy_sample():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -frames:v 2 sample.mp4"
    return get_sample(cmd)


class TestNvdecBasic(unittest.TestCase):
    def test_nvdec_no_file(self) -> None:
        """Does not crash when handling non-existing file"""
        with self.assertRaises(RuntimeError):
            _decode_video("lcbjbbvbhiibdctiljrnfvttffictefh.mp4")

    def test_nvdec_odd_size(self) -> None:
        """Odd width/height must be rejected"""
        dummy = _get_dummy_sample()
        with self.assertRaises(RuntimeError):
            _decode_video(dummy.path, scale_width=121)

        with self.assertRaises(RuntimeError):
            _decode_video(dummy.path, scale_height=257)

    def test_nvdec_negative(self) -> None:
        """Negative options must be rejected"""
        dummy = _get_dummy_sample()
        with self.assertRaises(RuntimeError):
            _decode_video(dummy.path, crop_left=-1)

        with self.assertRaises(RuntimeError):
            _decode_video(dummy.path, crop_top=-1)

        with self.assertRaises(RuntimeError):
            _decode_video(dummy.path, crop_right=-1)

        with self.assertRaises(RuntimeError):
            _decode_video(dummy.path, crop_bottom=-1)

    def test_nvdec_video_smoke_test(self) -> None:
        """Can decode video with NVDEC"""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 1000 sample.mp4"

        sample = get_sample(cmd)

        packets = spdl.io.demux_video(sample.path)
        print(packets)
        buffer = spdl.io.decode_packets_nvdec(
            packets,
            device_config=spdl.io.cuda_config(device_index=DEFAULT_CUDA),
        )

        tensor = spdl.io.to_torch(buffer)
        print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")
        self.assertEqual(tensor.shape[0], 1000)
        self.assertEqual(tensor.shape[1], 3)
        self.assertEqual(tensor.shape[2], 240)
        self.assertEqual(tensor.shape[3], 320)


def _save(array, prefix):
    from PIL import Image

    for i, arr in enumerate(array):
        Image.fromarray(arr[0]).save(f"{prefix}_{i}.png")


def split_nv12(array):
    w = array.shape[-1]
    h0 = array.shape[-2]
    h1, h2 = h0 * 2 // 3, h0 // 3
    y = array[:, :, :h1, :]
    uv = array[:, :, h1:, :].reshape(-1, 1, h2, w // 2, 2)
    u, v = uv[..., 0], uv[..., 1]
    return y, u, v


def _get_h264_sample():
    cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc,format=yuv420p -frames:v 100 sample.mp4"
    return get_sample(cmd)


class TestNvdecH264(unittest.TestCase):
    def test_nvdec_decode_h264_420p_basic(self) -> None:
        """NVDEC can decode YUV420P video."""
        h264 = _get_h264_sample()
        array = _decode_video(h264.path, timestamp=(0, 1.0))

        # y, u, v = split_nv12(array)
        # _save(array, "./base")
        # _save(y, "./base_y")
        # _save(u, "./base_u")
        # _save(v, "./base_v")

        self.assertEqual(array.dtype, torch.uint8)
        self.assertEqual(array.shape, (25, 3, 240, 320))

    # TODO: Test other formats like MJPEG, MPEG, HEVC, VC1 AV1 etc...

    def test_nvdec_decode_video_torch_allocator(self) -> None:
        """NVDEC can decode YUV420P video."""
        h264 = _get_h264_sample()
        allocator_called, deleter_called = False, False

        def allocator(size, device, stream):
            print("Calling allocator", flush=True)
            ptr = torch.cuda.caching_allocator_alloc(size, device, stream)
            nonlocal allocator_called
            allocator_called = True
            return ptr

        def deleter(ptr):
            print("Calling deleter", flush=True)
            torch.cuda.caching_allocator_delete(ptr)
            nonlocal deleter_called
            deleter_called = True

        def _test():
            self.assertFalse(allocator_called)
            self.assertFalse(deleter_called)
            array = _decode_video(
                h264.path,
                timestamp=(0, 1.0),
                allocator=(allocator, deleter),
            )
            self.assertTrue(allocator_called)
            self.assertEqual(array.dtype, torch.uint8)
            self.assertEqual(array.shape, (25, 3, 240, 320))

        _test()

        gc.collect()
        self.assertTrue(deleter_called)

    @unittest.expectedFailure  # FFmpeg seems to have issue with seeking HEVC. It returns 'Operation not permitted'. See https://trac.ffmpeg.org/ticket/9412
    def test_nvdec_decode_hevc_P010_basic(self) -> None:
        """NVDEC can decode HEVC video."""
        cmd = f"{FFMPEG_CLI} -hide_banner -y -f lavfi -i testsrc -t 3 -c:v libx265 -pix_fmt yuv420p10le -vtag hvc1 -y sample.hevc"
        sample = get_sample(cmd)

        array = _decode_video(
            sample.path,
            timestamp=(0, 1.0),
        )

        # testsrc default height is 240, so height = 240 + 240 // 2 = 360
        height = 360

        self.assertEqual(array.dtype, torch.uint8)
        self.assertEqual(array.shape, (25, 1, height, 320))

    def test_nvdec_decode_h264_420p_resize(self) -> None:
        """Check NVDEC decoder with resizing."""
        h264 = _get_h264_sample()
        width, height = 160, 120

        array = _decode_video(
            h264.path,
            timestamp=(0, 1.0),
            scale_width=width,
            scale_height=height,
        )

        # _save(array, "./resize")

        self.assertEqual(array.dtype, torch.uint8)
        self.assertEqual(array.shape, (25, 3, height, width))

    def test_nvdec_decode_h264_420p_crop(self) -> None:
        """Check NVDEC decoder with cropping."""
        h264 = _get_h264_sample()
        top, bottom, left, right = 40, 80, 100, 50
        h = 240 - top - bottom
        w = 320 - left - right

        rgb = _decode_video(
            h264.path,
            timestamp=(0, 1.0),
            crop_top=top,
            crop_bottom=bottom,
            crop_left=left,
            crop_right=right,
        )

        self.assertEqual(rgb.dtype, torch.uint8)
        self.assertEqual(rgb.shape, (25, 3, h, w))

        rgba0 = _decode_video(
            h264.path,
            timestamp=(0, 1.0),
        )

        for i in range(3):
            torch.testing.assert_close(
                rgb[:, i], rgba0[:, i, top : top + h, left : left + w]
            )

    def test_nvdec_decode_crop_resize(self) -> None:
        """Check NVDEC decoder with cropping and resizing."""
        h264 = _get_h264_sample()
        top, bottom, left, right = 40, 80, 100, 60
        h = (240 - top - bottom) // 2
        w = (320 - left - right) // 2

        array = _decode_video(
            h264.path,
            timestamp=(0.0, 1.0),
            crop_top=top,
            crop_bottom=bottom,
            crop_left=left,
            crop_right=right,
            scale_width=w,
            scale_height=h,
        )

        self.assertEqual(array.dtype, torch.uint8)
        self.assertEqual(array.shape, (25, 3, h, w))


def _is_ffmpeg4():
    vers = spdl.io.utils.get_ffmpeg_versions()
    print(vers)
    return vers["libavutil"][0] < 57


class TestColorConversion(unittest.TestCase):
    @unittest.skipIf(_is_ffmpeg4(), "FFmpeg4 is known to return a different result.")
    def test_color_conversion_rgba(self) -> None:
        """Providing pix_fmt="rgba" should produce (N,4,H,W) array."""
        # fmt: off
        cmd = f"""
        {FFMPEG_CLI} -hide_banner -y                    \
            -f lavfi -i color=color=0xff0000:size=32x64 \
            -f lavfi -i color=color=0x00ff00:size=32x64 \
            -f lavfi -i color=color=0x0000ff:size=32x64 \
            -filter_complex hstack=inputs=3             \
            -frames:v 25 sample.mp4
        """
        height, width = 64, 32
        sample = get_sample(cmd)

        array = _decode_video(sample.path, pix_fmt="rgb")

        self.assertEqual(array.dtype, torch.uint8)
        self.assertEqual(array.shape, (25, 3, height, 3 * width))

        # Red
        self.assertTrue(torch.all(array[:, 0, :, :width] == 255))
        self.assertTrue(torch.all(array[:, 1, :, :width] == 22))  # TODO: investivate if this is correct.
        self.assertTrue(torch.all(array[:, 2, :, :width] == 0))

        # Green
        self.assertTrue(torch.all(array[:, 0, :, width : 2 * width] == 0))
        self.assertTrue(torch.all(array[:, 1, :, width : 2 * width] == 217))  # TODO: investivate if this is correct.
        self.assertTrue(torch.all(array[:, 2, :, width : 2 * width] == 0))

        # Blue
        self.assertTrue(torch.all(array[:, 0, :, 2 * width :] == 0))
        self.assertTrue(torch.all(array[:, 1, :, 2 * width :] == 14))  # TODO: investivate if this is correct.
        self.assertTrue(torch.all(array[:, 2, :, 2 * width :] == 255))


def _create_mock_decoder() -> MagicMock:
    """Create a mock decoder with init and _reset methods."""
    mock_decoder = MagicMock()
    mock_decoder.init = MagicMock()
    mock_decoder._reset = MagicMock()
    return mock_decoder


def _get_test_data() -> tuple:
    """Get common test data: packets and cuda_config."""
    h264 = _get_h264_sample()
    packets = spdl.io.demux_video(h264.path)
    cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
    return packets, cuda_config


class TestNvdecThreadLocalCaching(unittest.TestCase):
    """Test the thread-local caching mechanism for NVDEC decoders."""

    def setUp(self) -> None:
        """Set up mocks for decoder creation."""
        # Clean up any cached decoders before each test
        spdl.io._core._del_cached_decoder()

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clean up any cached decoders after each test
        spdl.io._core._del_cached_decoder()

    @patch("spdl.io._core._libspdl_cuda._nvdec_decoder")
    def test_decoder_caching_same_thread(self, mock_nvdec_decoder: MagicMock) -> None:
        """Verify that nvdec_decoder returns the same cached instance within the same thread."""
        # Setup: Mock decoder creation to return a mock object
        mock_nvdec_decoder.return_value = _create_mock_decoder()
        packets, cuda_config = _get_test_data()

        # Execute: Create two decoders with use_cache=True
        decoder1 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=True)
        decoder2 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=True)

        # Assert: Both references should point to the same cached instance
        self.assertIs(
            decoder1, decoder2, "Decoders should be the same instance when cached"
        )
        # Verify decoder was created only once
        self.assertEqual(
            mock_nvdec_decoder.call_count, 1, "Decoder should be created only once"
        )

    @patch("spdl.io._core._libspdl_cuda._nvdec_decoder")
    def test_decoder_no_caching(self, mock_nvdec_decoder: MagicMock) -> None:
        """Verify that nvdec_decoder creates a new instance when use_cache=False."""
        # Setup: Mock decoder creation to return different mock objects each time
        mock_nvdec_decoder.side_effect = [
            _create_mock_decoder(),
            _create_mock_decoder(),
        ]
        packets, cuda_config = _get_test_data()

        # Execute: Create two decoders with use_cache=False
        decoder1 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=False)
        decoder2 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=False)

        # Assert: References should be different instances
        self.assertIsNot(
            decoder1, decoder2, "Decoders should be different instances when not cached"
        )
        # Verify decoder was created twice
        self.assertEqual(
            mock_nvdec_decoder.call_count, 2, "Decoder should be created twice"
        )

    @patch("spdl.io._core._libspdl_cuda._nvdec_decoder")
    def test_decoder_caching_different_threads(
        self, mock_nvdec_decoder: MagicMock
    ) -> None:
        """Verify that different threads get different decoder instances even when `use_cache=True`."""
        # Setup: Mock decoder creation to track creation count
        creation_count = {"count": 0}
        creation_lock = threading.Lock()

        def create_mock_decoder_with_count():
            with creation_lock:
                creation_count["count"] += 1
            return _create_mock_decoder()

        mock_nvdec_decoder.side_effect = create_mock_decoder_with_count
        packets, cuda_config = _get_test_data()

        decoder_refs = []

        def get_decoder_in_thread():
            decoder = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=True)
            decoder_refs.append(id(decoder))

        # Execute: Create decoders in two different threads
        thread1 = threading.Thread(target=get_decoder_in_thread)
        thread2 = threading.Thread(target=get_decoder_in_thread)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Assert: Different threads should get different decoder instances
        self.assertEqual(
            len(decoder_refs), 2, "Should have captured two decoder references"
        )
        self.assertNotEqual(
            decoder_refs[0],
            decoder_refs[1],
            "Different threads should have different decoder instances",
        )
        # Verify decoder was created twice (once per thread)
        self.assertEqual(
            creation_count["count"], 2, "Decoder should be created once per thread"
        )

    @patch("spdl.io._core._libspdl_cuda._nvdec_decoder")
    def test_decoder_cache_cleared_on_crop(self, mock_nvdec_decoder: MagicMock) -> None:
        """Verify that providing crop parameters forces recreation of the decoder.

        # Note this is not a spec. It's just the way the implementation is.
        # We prefer to keep cache even when cropping is provided, but we have not figured
        # out the way yet.
        # Feel free to delete this test if you figured.
        """

        # Setup: Mock decoder creation to return different mock objects each time
        mock_nvdec_decoder.side_effect = [
            _create_mock_decoder(),
            _create_mock_decoder(),
        ]
        packets, cuda_config = _get_test_data()

        # Execute: Create a decoder without crop, then with crop (use_cache is True but should be ignored)
        decoder1 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=True)
        decoder2 = spdl.io.nvdec_decoder(  # noqa: F841
            cuda_config,
            packets.codec,
            use_cache=True,
            crop_left=10,
        )

        # Assert: Should be different instances because crop forces recreation
        self.assertIsNot(
            decoder1, decoder2, "Crop parameters should force decoder recreation"
        )
        # Verify decoder was created twice (crop forces recreation)
        self.assertEqual(
            mock_nvdec_decoder.call_count,
            2,
            "Decoder should be recreated when crop parameters change",
        )

    @patch("spdl.io._core._libspdl_cuda._nvdec_decoder")
    def test_cache_cleanup_with_hasattr_delattr(
        self, mock_nvdec_decoder: MagicMock
    ) -> None:
        """Verify that cache cleanup works correctly with hasattr/delattr pattern."""
        # Setup: Mock decoder creation
        mock_nvdec_decoder.return_value = _create_mock_decoder()
        packets, cuda_config = _get_test_data()

        # Execute: Create decoder, delete cache, create again
        decoder1 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=True)  # noqa: F841
        spdl.io._core._del_cached_decoder()
        decoder2 = spdl.io.nvdec_decoder(cuda_config, packets.codec, use_cache=True)  # noqa: F841

        # Assert: Second decoder should be a different instance after cache clear
        # Note: Due to mocking, they'll have the same mock object, but verify creation count
        self.assertEqual(
            mock_nvdec_decoder.call_count,
            2,
            "Decoder should be created twice after cache clear",
        )

    @patch("spdl.io._core._libspdl_cuda._nvdec_decoder")
    def test_thread_local_isolation_with_getattr(
        self, mock_nvdec_decoder: MagicMock
    ) -> None:
        """Verify getattr properly handles thread-local storage without AttributeError."""
        # Setup: Mock decoder creation to return unique mocks
        mock_nvdec_decoder.side_effect = lambda: _create_mock_decoder()
        packets, cuda_config = _get_test_data()

        results = []
        errors = []

        def get_decoder_in_thread():
            try:
                # This tests that getattr() works correctly even when _decoder doesn't exist yet
                decoder = spdl.io.nvdec_decoder(
                    cuda_config, packets.codec, use_cache=True
                )
                results.append(id(decoder))
            except AttributeError as e:
                errors.append(f"AttributeError: {e}")
            except Exception as e:
                errors.append(f"Unexpected error: {e}")

        # Execute: Create decoders in multiple threads to test getattr behavior
        threads = [threading.Thread(target=get_decoder_in_thread) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Assert: All threads should complete without AttributeError
        self.assertEqual(
            len(errors),
            0,
            f"getattr() should handle missing _decoder attribute gracefully: {errors}",
        )
        self.assertEqual(len(results), 3, "All threads should create decoders")
        # Verify each thread got its own decoder instance
        self.assertEqual(
            len(set(results)), 3, "Each thread should have a unique decoder instance"
        )


class TestDecodeAll(unittest.TestCase):
    """Tests for the decode_all method (batch decoding with pre-allocated buffers)."""

    def test_decode_all_outputs_expected_data(self) -> None:
        """Verify decode_all outputs data with expected shape and dtype."""
        # Setup: Create test video sample
        h264 = _get_h264_sample()
        cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)
        packets = spdl.io.demux_video(h264.path, timestamp=(0, 1.0))

        # Apply BSF for H264
        packets = spdl.io.apply_bsf(packets, "h264_mp4toannexb")

        # Execute: Use nvdec_decoder with decode_all
        decoder = spdl.io.nvdec_decoder(cuda_config, packets.codec)
        nv12_buffer = decoder.decode_all(packets)

        # Convert to torch tensor to verify shape and dtype
        tensor = spdl.io.to_torch(nv12_buffer)

        # Assert: Verify expected data format
        # decode_all returns NV12 buffer with shape [num_frames, h*1.5, width]
        # For testsrc default 320x240: h*1.5 = 360
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertEqual(len(tensor.shape), 3)  # [num_frames, h*1.5, width]
        self.assertEqual(tensor.shape[0], 25)  # 25 frames for 1 second at 25fps
        self.assertEqual(tensor.shape[1], 360)  # 240 * 1.5 = 360 (NV12 height)
        self.assertEqual(tensor.shape[2], 320)  # testsrc default width

    def test_decode_all_matches_regular_decode(self) -> None:
        """Verify decode_all produces the same result as decode() + flush()."""
        # Setup: Create test video sample
        h264 = _get_h264_sample()
        cuda_config = spdl.io.cuda_config(device_index=DEFAULT_CUDA)

        # Demux the same video twice for separate decoding paths
        packets1 = spdl.io.demux_video(h264.path, timestamp=(0, 1.0))
        packets2 = spdl.io.demux_video(h264.path, timestamp=(0, 1.0))

        # Apply BSF for H264
        packets1 = spdl.io.apply_bsf(packets1, "h264_mp4toannexb")
        packets2 = spdl.io.apply_bsf(packets2, "h264_mp4toannexb")

        # Execute: Decode using regular decode + flush
        decoder1 = spdl.io.nvdec_decoder(cuda_config, packets1.codec, use_cache=False)
        regular_buffers = decoder1.decode(packets1)
        regular_buffers += decoder1.flush()

        # Convert regular output to comparable format
        # nv12_to_planar_rgb expects list of 2D NV12 buffers
        regular_rgb = spdl.io.lib._libspdl_cuda.nv12_to_planar_rgb(
            regular_buffers, device_config=cuda_config
        )
        regular_tensor = spdl.io.to_torch(regular_rgb)

        # Execute: Decode using decode_all
        decoder2 = spdl.io.nvdec_decoder(cuda_config, packets2.codec, use_cache=False)
        batch_buffer = decoder2.decode_all(packets2)
        num_frames = batch_buffer.__cuda_array_interface__["shape"][0]

        # Convert batch output to RGB using batched conversion
        batch_rgb = spdl.io.lib._libspdl_cuda.nv12_to_planar_rgb_batched(
            batch_buffer, device_config=cuda_config
        )
        batch_tensor = spdl.io.to_torch(batch_rgb)

        # Assert: Both methods should produce the same output
        self.assertEqual(regular_tensor.shape, batch_tensor.shape)
        self.assertEqual(regular_tensor.dtype, batch_tensor.dtype)

        # Compare actual pixel values
        torch.testing.assert_close(regular_tensor, batch_tensor)
