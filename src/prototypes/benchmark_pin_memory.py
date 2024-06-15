# pyre-ignore-all-errors

import time

import spdl.io


src = "/home/moto/sample.mp4"

packets = spdl.io.demux_video(src)
frames = spdl.io.decode_packets(packets, filter_desc="scale=224:224,format=rgb24")


def _test(pin_memory, batch_size, num_batches=500):
    input_frames = [frames[:batch_size] for _ in range(num_batches)]

    t0 = time.monotonic()
    for input_frame_set in input_frames:
        buffer = spdl.io.convert_frames(input_frame_set, pin_memory=pin_memory)
        buffer = spdl.io.transfer_buffer(
            buffer, cuda_config=spdl.io.cuda_config(device_index=0)
        )
    elapsed = time.monotonic() - t0

    print(
        f"{pin_memory=:1}, {batch_size=:2d}, {num_batches=:3d}: {elapsed:.2f} seconds"
    )


# warm-up
_test(True, 10)


for batch_size in [32, 64]:
    for pin_memory in [False, True]:
        _test(pin_memory, batch_size)
