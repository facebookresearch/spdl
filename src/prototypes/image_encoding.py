import asyncio
import logging
from pathlib import Path

import numpy as np

import spdl.io
from spdl.dataloader import BackgroundTaskExecutor

_LG = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(filename)s %(lineno)s: %(message)s"
)

root = Path.home() / "tmp"

"""
coros = []

enc_cfg = spdl.io.encode_config(width=128, height=96, scale_algo="neighbor")
data = np.random.randint(255, size=(16, 16, 3), dtype=np.uint8)
coro = spdl.io.async_encode_image(
    str(root / "foo.np.png"),
    data,
    pix_fmt="rgb24",
    encode_config=enc_cfg,
)
coros.append(coro)

data = torch.from_numpy(data)
coro = spdl.io.async_encode_image(
    str(root / "foo.torch.png"),
    data,
    pix_fmt="rgb24",
    encode_config=enc_cfg,
)
coros.append(coro)

data = np.random.randint(255, size=(16, 16), dtype=np.uint8)
coro = spdl.io.async_encode_image(
    str(root / "foo.np.gray.png"),
    data,
    pix_fmt="gray8",
    encode_config=enc_cfg,
)
coros.append(coro)

data = torch.from_numpy(data)
coro = spdl.io.async_encode_image(
    str(root / "foo.torch.gray.png"),
    data,
    pix_fmt="gray8",
    encode_config=enc_cfg,
)
coros.append(coro)

data = data.cuda()
coro = spdl.io.async_encode_image(
    str(root / "foo.torch.cuda.gray.png"),
    data,
    pix_fmt="gray8",
    encode_config=enc_cfg,
)
coros.append(coro)
"""


async def batch_save(batch, pix_fmt):
    tasks = []
    for i in range(batch.shape[0]):
        print(i)
        path = str(root / f"foo.torch.batch_{pix_fmt}_{i}.png")
        coro = spdl.io.async_encode_image(path, batch[i], pix_fmt=pix_fmt)
        tasks.append(asyncio.create_task(coro))
        break

    await asyncio.wait(tasks)


with BackgroundTaskExecutor() as queue:
    data = np.random.randint(255, size=(32, 24, 16, 3), dtype=np.uint8)
    queue.put(batch_save(data, "rgb24"))

    data = np.random.randint(255, size=(32, 24, 16), dtype=np.uint8)
    queue.put(batch_save(data, "gray8"))

    data = np.random.randint(255, size=(32, 3, 24, 16), dtype=np.uint8)
    queue.put(batch_save(data, "yuv444p"))

    _LG.info(f"{queue.qsize()=}")
