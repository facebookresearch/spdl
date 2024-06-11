import asyncio
import time
from asyncio import Task

import spdl.io
import spdl.utils

import torch
from spdl.dataloader import BackgroundGenerator
from spdl.io import ImageFrames
from spdl.utils import run_async
from torch import Tensor
from torch.profiler import profile

from torch.utils.data import DataLoader, IterableDataset
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


def _to_aiter(iterable, batch_size, collate_fn, drop_last):

    async def _aiter():
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield await collate_fn(batch)
                batch = []
        if not drop_last and batch:
            yield await collate_fn(batch)

    return aiter(_aiter())


class _DataSet(IterableDataset):
    def __init__(
        self,
        dataset,
        collate_fn,
        batch_size: int,
        num_workers: int,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        timeout: int | float = 30,
    ) -> None:
        self._dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout

    def __iter__(self):
        gen = _to_aiter(self._dataset, self.batch_size, self.collate_fn, self.drop_last)
        bgg = BackgroundGenerator(
            gen,
            num_workers=self.num_workers,
            queue_size=self.prefetch_factor,
            timeout=self.timeout,
        )
        return iter(bgg)


def _dataloader(dataset, collate_fn, batch_size, num_workers, prefetch_factor=8):
    _dataset = _DataSet(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return DataLoader(
        _dataset,
        batch_size=None,
        num_workers=0,
    )


def _main():
    dataloader = _dataloader(
        ImageNet(root=root, loader=lambda x: x),
        collate_fn=_collate_fn,
        batch_size=32,
        num_workers=8,
    )

    trace_path = "/home/moto/tmp/trace/dataloader"
    with (
        profile() as prof,
        spdl.utils.tracing(f"{trace_path}.pftrace"),
    ):
        t0 = time.monotonic()
        num_frames = 0
        try:
            for i, (batch, classes) in enumerate(dataloader):
                print(batch.shape, batch.dtype, classes.shape, classes.dtype)
                num_frames += batch.shape[0]

                if i == 300:
                    break
        finally:
            elapsed = time.monotonic() - t0
            print(f"{elapsed:.2f} [sec], {num_frames=}")
    prof.export_chrome_trace(f"{trace_path}.json")


if __name__ == "__main__":
    _main()
