import asyncio

import numpy as np

import pytest

import spdl.io
import spdl.utils
from spdl.io import get_audio_filter_desc, get_video_filter_desc
from spdl.lib import _libspdl


def test_failure():
    """Demuxer fails without segfault if the input does not exist"""

    with pytest.raises(RuntimeError, match="Failed to open the input"):
        spdl.io.Demuxer("dvkgviuerehidguburuekkhgjijfjbkj")


async def _decode_packet(packets):
    frames = await spdl.io.async_decode_packets(packets)
    print(frames)
    buffer = await spdl.io.async_convert_frames(frames)
    print(buffer)
    array = spdl.io.to_numpy(buffer)
    print(array.shape, array.dtype)
    return array


async def _test_async_decode(demux_fn, timestamps):
    # There was a case where the underlying file device was delayed, and the
    # generated sample file is not ready when the test is started, so
    # sleeping here for 1 second to make sure the file is ready.
    await asyncio.sleep(1)

    tasks = []
    for timestamp in timestamps:
        packets = demux_fn(timestamp)
        print(packets)
        tasks.append(asyncio.create_task(_decode_packet(packets)))

    return await asyncio.gather(*tasks)


def test_decode_audio_clips(get_sample):
    """Can decode audio clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test():
        timestamps = [(i, i + 1) for i in range(2)]
        demuxer = spdl.io.Demuxer(sample.path)
        arrays = await _test_async_decode(demuxer.demux_audio, timestamps)

        assert len(arrays) == 2
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (48000, 1)
            assert arr.dtype == np.int16

    asyncio.run(_test())


def test_decode_audio_clips_num_frames(get_sample):
    """Can decode audio clips with padding/dropping."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=16000:duration=1' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _decode(src, num_frames=None):
        with spdl.io.Demuxer(src) as demuxer:
            packets = demuxer.demux_audio(window=(0, 1))
            filter_desc = get_audio_filter_desc(timestamp=(0, 1), num_frames=num_frames)
            frames = await spdl.io.async_decode_packets(
                packets, filter_desc=filter_desc
            )
            buffer = await spdl.io.async_convert_frames(frames)
            return spdl.io.to_numpy(buffer)

    async def _test(src):
        arr0 = await _decode(src)
        assert arr0.dtype == np.int16
        assert arr0.shape == (16000, 1)

        num_frames = 8000
        arr1 = await _decode(src, num_frames=num_frames)
        assert arr1.dtype == np.int16
        assert arr1.shape == (num_frames, 1)
        assert np.all(arr1 == arr0[:num_frames])

        num_frames = 32000
        arr2 = await _decode(src, num_frames=num_frames)
        assert arr2.dtype == np.int16
        assert arr2.shape == (num_frames, 1)
        assert np.all(arr2[:16000] == arr0)
        assert np.all(arr2[16000:] == 0)

    asyncio.run(_test(sample.path))


def test_decode_video_clips(get_sample):
    """Can decode video clips."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)
    N = 10

    async def _test():
        timestamps = [(i, i + 1) for i in range(N)]
        demuxer = spdl.io.Demuxer(sample.path)
        arrays = await _test_async_decode(demuxer.demux_video, timestamps)
        assert len(arrays) == N
        for i, arr in enumerate(arrays):
            print(i, arr.shape, arr.dtype)
            assert arr.shape == (25, 240, 320, 3)
            assert arr.dtype == np.uint8

    asyncio.run(_test())


def test_decode_video_clips_num_frames(get_sample):
    """Can decode video clips with padding/dropping."""
    if "tpad" not in spdl.utils.get_ffmpeg_filters():
        raise pytest.skip("tpad filter is not available. Install FFmepg >= 4.2.")

    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 50 sample.mp4"
    sample = get_sample(cmd)

    async def _decode(src, pix_fmt="rgb24", **kwargs):
        with spdl.io.Demuxer(src) as demuxer:
            packets = demuxer.demux_video(window=(0, 2))
            filter_desc = get_video_filter_desc(
                timestamp=(0, 2), pix_fmt=pix_fmt, **kwargs
            )
            frames = await spdl.io.async_decode_packets(
                packets, filter_desc=filter_desc
            )
            buffer = await spdl.io.async_convert_frames(frames)
            return spdl.io.to_numpy(buffer)

    async def _test(src):
        arr0 = await _decode(src)
        assert arr0.dtype == np.uint8
        assert arr0.shape == (50, 240, 320, 3)

        num_frames = 25
        arr1 = await _decode(src, num_frames=num_frames)
        assert arr1.dtype == np.uint8
        assert arr1.shape == (num_frames, 240, 320, 3)
        assert np.all(arr1 == arr0[:num_frames])

        num_frames = 100
        arr2 = await _decode(src, num_frames=num_frames)
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == arr2[50])

        num_frames = 100
        arr2 = await _decode(src, num_frames=num_frames, pad_mode="black")
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == 0)

        num_frames = 100
        arr2 = await _decode(src, num_frames=num_frames, pad_mode="white")
        assert arr2.dtype == np.uint8
        assert arr2.shape == (num_frames, 240, 320, 3)
        assert np.all(arr2[:50] == arr0)
        assert np.all(arr2[50:] == 255)

    asyncio.run(_test(sample.path))


def test_decode_video_frame_rate_pts(get_sample):
    """Applying frame rate outputs correct PTS."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -r 10 -frames:v 20 sample.mp4"
    sample = get_sample(cmd)

    async def _test(src):
        packets = await spdl.io.async_demux_video(src)
        frames_ref = await spdl.io.async_decode_packets(packets.clone())
        frames = await spdl.io.async_decode_packets(
            packets, filter_desc=get_video_filter_desc(frame_rate=(5, 1))
        )

        pts_ref = frames_ref._get_pts()
        pts = frames._get_pts()
        print(pts_ref, pts)

        assert np.all(pts_ref[::2] == pts)

    asyncio.run(_test(sample.path))


async def _decode_image(path):
    packets = await spdl.io.async_demux_image(path)
    print(packets)
    frames = await spdl.io.async_decode_packets(packets)
    print(frames)
    assert type(frames) is _libspdl.FFmpegImageFrames
    return frames


async def _batch_decode_image(paths):
    return await asyncio.gather(*[_decode_image(path) for path in paths])


def test_decode_image(get_sample):
    """Can decode an image."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        buffer = await spdl.io.async_load_image(src)
        array = spdl.io.to_numpy(buffer)
        print(array.shape, array.dtype)
        assert array.dtype == np.uint8
        assert array.shape == (240, 320, 3)

    asyncio.run(_test(sample.path))


def test_batch_decode_image(get_samples):
    """Can decode a batch of images."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 250 sample_%03d.jpg"
    samples = get_samples(cmd)

    flist = ["NON_EXISTING_FILE.JPG"] + samples

    async def _test():
        buffer = await spdl.io.async_load_image_batch(
            flist, width=None, height=None, pix_fmt=None, strict=False
        )
        assert buffer.__array_interface__["shape"] == (250, 3, 240, 320)

    asyncio.run(_test())


def test_async_convert_audio(get_sample):
    """async_convert_frames can convert FFmpegAudioFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i 'sine=frequency=1000:sample_rate=48000:duration=3' -c:a pcm_s16le sample.wav"
    sample = get_sample(cmd)

    async def _test(src):
        ts = [(1, 2)]
        demuxer = spdl.io.Demuxer(src)
        arrays = await _test_async_decode(demuxer.demux_audio, ts)
        array = arrays[0]
        print(array.dtype, array.shape)
        assert array.shape == (48000, 1)

    asyncio.run(_test(sample.path))


def test_async_convert_video(get_sample):
    """async_convert_frames can convert FFmpegVideoFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1000 sample.mp4"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        packets = await spdl.io.async_demux_video(src)
        frames = await spdl.io.async_decode_packets(packets)
        buffer = await spdl.io.async_convert_frames(frames)
        array = spdl.io.to_numpy(buffer)
        print(array.dtype, array.shape)
        assert array.shape == (1000, 240, 320, 3)

    asyncio.run(_test(sample.path))


def test_async_convert_image(get_sample):
    """async_convert_frames can convert FFmpegImageFrames to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd, width=320, height=240)

    async def _test(src):
        frames = await _decode_image(src)
        buffer = await spdl.io.async_convert_frames(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (240, 320, 3)

    asyncio.run(_test(sample.path))


def test_async_convert_batch_image(get_samples):
    """async_convert_frames can convert list[FFmpegImageFrames] to Buffer"""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 4 sample_%03d.jpg"
    flist = get_samples(cmd)

    async def _test(flist):
        frames = await _batch_decode_image(flist)
        buffer = await spdl.io.async_convert_frames(frames)
        print(buffer)
        arr = spdl.io.to_numpy(buffer)
        print(arr.dtype, arr.shape)
        assert arr.shape == (4, 240, 320, 3)

    asyncio.run(_test(flist))
