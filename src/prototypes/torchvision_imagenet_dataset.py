import asyncio
from asyncio import Task

import spdl.io
import spdl.utils

import torch
from spdl.io import ImageFrames
from spdl.utils import run_async
from torch import Tensor
from pytorch_spdl.dataloader import DataLoader
from torchvision.datasets.imagenet import ImageNet

root = "/home/moto/local/imagenet"


def _decode(path: str, width=224, height=224, pix_fmt="rgb24") -> ImageFrames:
    packets = spdl.io.demux_image(path)
    return spdl.io.decode_packets(
        packets,
        filter_desc=spdl.io.get_video_filter_desc(
            scale_width=224, scale_height=224, pix_fmt="rgb24"
        ),
    )


def _async_decode(path: str) -> Task[ImageFrames]:
    coro = run_async(_decode, path)
    return asyncio.create_task(coro)


async def _collate_fn(samples: list[tuple[str, int]]) -> tuple[Tensor, Tensor]:
    tasks = [_async_decode(s[0]) for s in samples]
    classes_ = [s[1] for s in samples]

    await asyncio.wait(tasks)

    frames = []
    for task in tasks:
        try:
            frame = task.result()
        except Exception as e:
            print("Failed to decode image:", e)
        else:
            frames.append(frame)

    buffer = await spdl.io.async_convert_frames(frames)
    return spdl.io.to_torch(buffer), torch.tensor(classes_)


def _main():
    dataset = ImageNet(root=root, loader=lambda x: x)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=3,
        collate_fn=_collate_fn,
        prefetch_factor=2,
    )

    with spdl.utils.tracing("/home/moto/tmp/trace/dataloader.pftrace"):
        for i, (batch, classes) in enumerate(dataloader):
            print(batch.shape, batch.dtype, classes.shape, classes.dtype)

            if i == 10:
                break


if __name__ == "__main__":
    _main()
