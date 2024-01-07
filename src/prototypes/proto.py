import logging

from spdl import libspdl


def _parse_python_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-i", "--input-video", help="Input video file.", required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("others", nargs="*")
    return parser.parse_args()


n_plot = 0


def _plot(frames):
    import matplotlib.pyplot as plt

    global n_plot
    for i, frame in enumerate(frames):
        plt.imshow(frame)
        plt.savefig(f"tmp/frame_{n_plot}_{i:03d}.png")

    n_plot += 1


def _plot_audio(frames, sample_rate):
    import matplotlib.pyplot as plt

    print(f"{frames.shape=}")
    fig, axes = plt.subplots(2 * frames.shape[0])
    for i, channel in enumerate(frames):
        print(
            f"{channel.shape=}, {channel.dtype=}, ({channel.min()=}, {channel.max()=})"
        )
        axes[2 * i].specgram(channel, Fs=sample_rate)
        axes[2 * i + 1].plot(channel)

    plt.show()


def _main():
    args = _parse_python_args()
    _init_logging(args.debug)
    libspdl.init_folly(args.others)
    if args.debug:
        libspdl.set_ffmpeg_log_level(40)

    # test_video(args)
    test_audio(args)


def test_audio(args):
    src = args.input_video
    src2 = f"mmap://{src}"

    configs = [
        {
            "src": src,
            "timestamps": [(5.0, 10.0)],
            "sample_rate": 8000,
        },
        {
            "src": src2,
            "timestamps": [(5.0, 10.0)],
            "sample_rate": 8000,
            "sample_fmt": "s16p",
        },
    ]

    for i, cfg in enumerate(configs):
        print("*" * 40)
        print(cfg)
        print("*" * 40)

        decoded_frames = libspdl.decode_audio(**cfg)[0]
        print(f"{len(decoded_frames)=}")
        print(f"{decoded_frames.num_samples=}")
        sliced = decoded_frames[2:7:2]
        print(len(sliced))
        del sliced

        a = libspdl.to_numpy(decoded_frames, index=None, format="channel_first")
        print(f"{a.shape=}, {a.dtype=}")

        if args.plot:
            _plot_audio(a, decoded_frames.sample_rate)


def test_video(args):
    src = args.input_video
    src2 = f"mmap://{src}"

    configs = [
        {
            "src": src,
            "timestamps": [(0.0, 1.0), (10.0, 11.0)],
        },
        {
            "src": src2,
            "timestamps": [(0.0, 1.0), (10.0, 11.0)],
            "frame_rate": "30000/1001",
            "width": 36,
            "height": 48,
            "pix_fmt": "rgb24",
        },
    ]

    if args.gpu:
        for i in range(2):
            libspdl.create_cuda_context(i, use_primary_context=True)

        import torch

        a = torch.empty([1, 2]).to(device=f"cuda:{i}")
        print(a)

    for cfg in configs:
        if args.gpu:
            cfg["decoder"] = "h264_cuvid"
            cfg["cuda_device_index"] = 1
            cfg["width"] = None
            cfg["height"] = None
            cfg["pix_fmt"] = None
            cfg["frame_rate"] = None

        print("*" * 40)
        print(cfg)
        print("*" * 40)

        decoded_frames = libspdl.decode_video(**cfg)
        for frames in decoded_frames:
            print(
                f"{frames.format=}, {len(frames)=}, {frames.width=}, {frames.height=}"
            )
            frames = frames[::2]
            print(
                f"{frames.format=}, {len(frames)=}, {frames.width=}, {frames.height=}"
            )

            a = libspdl.to_numpy(frames, format="NHWC")
            print(a.shape, a.dtype)
            if args.plot:
                _plot(a)

            for p in range(frames.num_planes):
                a = libspdl.to_numpy(frames, index=p, format="NHWC")
                print(a.shape, a.dtype)
                if args.plot:
                    _plot(a)


def _init_logging(debug):
    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.getLogger("spdl").setLevel(logging.DEBUG)


if __name__ == "__main__":
    _main()
