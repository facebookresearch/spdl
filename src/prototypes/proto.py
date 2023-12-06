import spdl.lib._spdl_ffmpeg

src = "/home/moto/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
src2 = f"mmap://{src}"

engine = spdl.lib._spdl_ffmpeg.Engine(3, 6, 10)
engine.enqueue(src, [3.])
engine.enqueue(src2, [3.])
engine.dequeue()
engine.dequeue()
