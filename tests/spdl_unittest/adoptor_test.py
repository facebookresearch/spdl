import asyncio

import numpy as np
import spdl
import spdl.libspdl


def test_demux_image_bytes(get_sample):
    """Image (gray) can be decoded from bytes."""
    cmd = "ffmpeg -hide_banner -y -f lavfi -i color=0x000000,format=gray -frames:v 1 sample.png"
    sample = get_sample(cmd, width=320, height=240)

    async def _decode(src, adoptor=None):
        packets = await spdl.async_demux_image(src, adoptor=adoptor)
        print(packets)
        frames = await spdl.async_decode(packets)
        print(frames)
        buffer = await spdl.async_convert_image(frames)
        return np.array(buffer, copy=False)

    async def _test(path):
        ref = await _decode(path)
        with open(sample.path, "rb") as f:
            buffer = f.read()
        hyp = await _decode(buffer, spdl.libspdl.BytesAdoptor())

        assert hyp.shape == (1, 240, 320)
        assert np.all(ref == hyp)

    asyncio.run(_test(sample.path))
