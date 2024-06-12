import gc
import time

import spdl.io
import spdl.utils

import torch
from pytorch_spdl.dataloader import DataLoader
from spdl.io import ImageFrames
from spdl.lib import _libspdl
from torch import Tensor
from torch.profiler import profile

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


def _batch_decode(samples: list[tuple[str, int]]) -> tuple[Tensor, Tensor]:
    classes_ = torch.tensor([cls_ for _, cls_ in samples])

    frames = [_decode(path) for path, _ in samples]
    buffer = spdl.io.convert_frames(frames)
    buffer = spdl.io.transfer_buffer(
        buffer,
        cuda_config=spdl.io.cuda_config(
            device_index=0,
            allocator=(
                torch.cuda.caching_allocator_alloc,
                torch.cuda.caching_allocator_delete,
            ),
        ),
    )
    return spdl.io.to_torch(buffer), classes_


class _record_gc:
    def __init__(self):
        self.record = None

    def __call__(self, phase, info):
        if phase == "start":
            # self.record = torch.ops.profiler._record_function_enter_new("gc", None)
            _libspdl.trace_event_begin("gc")
        else:
            # torch.ops.profiler._record_function_exit(self.record)
            _libspdl.trace_event_end()


gc.callbacks.append(_record_gc())
spdl.utils.set_ffmpeg_log_level(0)


def _main():
    dataloader = DataLoader(
        ImageNet(root=root, loader=lambda x: x),
        collate_fn=_batch_decode,
        batch_size=32,
        num_workers=124,
        prefetch_factor=2,
    )

    dataloader = iter(dataloader)
    for _ in range(50):
        next(dataloader)

    trace_path = "/home/moto/tmp/trace/dataloader"
    with (
        profile() as prof,
        spdl.utils.tracing(f"{trace_path}.pftrace", buffer_size=4096 * 16),
    ):
        t0 = time.monotonic()
        num_frames = 0
        try:
            for i, (batch, classes) in enumerate(dataloader):
                # print(batch.shape, batch.dtype, classes.shape, classes.dtype)
                num_frames += batch.shape[0]

                if i == 499:
                    break

        finally:
            elapsed = time.monotonic() - t0
            QPS = num_frames / elapsed
            print(f"{QPS=:.2f}: {elapsed:.2f} [sec], {num_frames=}")
    prof.export_chrome_trace(f"{trace_path}.json")


if __name__ == "__main__":
    _main()
