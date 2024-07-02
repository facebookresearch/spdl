import asyncio

import spdl.io


def test_image_frame_metadata(get_sample):
    """Smoke test for image frame metadata.
    Ideally, we should use images with EXIF data, but ffmpeg
    does not seem to support exif, and I don't want to check-in
    assets data, so just smoke test.
    """
    cmd = "ffmpeg -hide_banner -y -f lavfi -i testsrc -frames:v 1 sample.jpg"
    sample = get_sample(cmd)

    async def test(src):
        packets = await spdl.io.async_demux_image(src)
        frames = await spdl.io.async_decode_packets(packets)

        assert frames.metadata == {}

    asyncio.run(test(sample.path))
