def get_src_video():
    return "NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"


def get_video_frames(src, pix_fmt="rgb24", height=128, width=256):
    from spdl import libspdl

    engine = libspdl.Engine(10)
    engine.enqueue(
        src=src,
        timestamps=[0.0],
        pix_fmt=pix_fmt,
        height=height,
        width=width,
    )
    return engine.dequeue()
