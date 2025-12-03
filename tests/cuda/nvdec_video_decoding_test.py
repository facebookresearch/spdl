# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest

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
