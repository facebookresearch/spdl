"""Benchmark DALI's batch image decoding performance"""

from timeit import default_timer as timer

import nvidia.dali.fn as fn

import nvidia.dali.types as types
from nvidia.dali import Pipeline, pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def _parse_args(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="The root directory where the images are stored.",
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="The number of workers (GPUs) to use.",
    )
    parser.add_argument(
        "--num-threads",
        default=8,
        type=int,
        help="The number of threads used by pipeline.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
    )
    return parser.parse_args(args)


def _benchmark(data_dir, batch_size, num_threads, num_workers):
    @pipeline_def
    def _pipeline(shard_id, num_shards):
        files, labels = fn.readers.file(
            file_root=data_dir,
            shard_id=Pipeline.current().device_id,
            num_shards=num_shards,
            name="Reader",
        )
        images = fn.decoders.image(files, device="cpu", output_type=types.RGB)
        images = fn.resize(images, resize_x=256, resize_y=256)
        return images.gpu(), labels.gpu()

    pipes = [
        _pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            shard_id=device_id,
            num_shards=num_workers,
        )
        for device_id in range(num_workers)
    ]
    iterator = DALIGenericIterator(pipes, ["data", "label"], reader_name="Reader")

    num_frames, num_batches = 0, 0
    t_start = timer()
    for items in iterator:
        assert len(items) == num_workers
        for i, item in enumerate(items):
            batch = item["data"]
            assert batch.is_contiguous()
            assert batch.is_cuda
            assert batch.device.index == i

            # print(type(batch))
            num_frames += len(batch)
            num_batches += 1

    elapsed = timer() - t_start
    fps = num_frames / elapsed
    print(f"{num_threads=}\t{fps=}\t{num_batches}\t{num_frames}\t{elapsed=}")


def _main():
    args = _parse_args()
    print(args, flush=True)
    _benchmark(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    _main()
