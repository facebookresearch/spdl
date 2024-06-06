from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, overload

from numpy.typing import ArrayLike


class AudioPackets:
    def __repr__(self) -> str: ...

    @property
    def timestamp(self) -> tuple[float, float] | None: ...

    def clone(self) -> AudioPackets: ...

class BytesAdaptor:
    def __init__(self) -> None: ...

class CPUBuffer:
    @property
    def __array_interface__(self) -> dict: ...

class CUDABuffer:
    @property
    def __cuda_array_interface__(self) -> None: ...

    @property
    def device_index(self) -> None: ...

class CUDAConfig:
    def __init__(self, device_index: int, stream: int = 0, allocator: tuple[Callable[[int, int, int], int], Callable[[int], None]] | None = None) -> None: ...

class DecodeConfig:
    def __init__(self, decoder: str | None = None, decoder_options: Mapping[str, str] | None = None) -> None: ...

class DemuxConfig:
    def __init__(self, format: str | None = None, format_options: Mapping[str, str] | None = None, buffer_size: int = 8096) -> None: ...

class EncodeConfig:
    def __init__(self, muxer: str | None = None, muxer_options: Mapping[str, str] | None = None, encoder: str | None = None, encoder_options: Mapping[str, str] | None = None, format: str | None = None, width: int = -1, height: int = -1, scale_algo: str | None = None, filter_desc: str | None = None, bit_rate: int = -1, compression_level: int = -1, qscale: int = -1, gop_size: int = -1, max_bframes: int = -1) -> None: ...

class FFmpegAudioFrames:
    @property
    def num_frames(self) -> int: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def num_channels(self) -> int: ...

    @property
    def format(self) -> str: ...

    def __len__(self) -> int: ...

    def __repr__(self) -> str: ...

    def clone(self) -> FFmpegAudioFrames: ...

class FFmpegImageFrames:
    @property
    def num_planes(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def format(self) -> str: ...

    def __repr__(self) -> str: ...

    def clone(self) -> FFmpegImageFrames: ...

    @property
    def pts(self) -> float: ...

class FFmpegVideoFrames:
    @property
    def num_frames(self) -> int: ...

    @property
    def num_planes(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def format(self) -> str: ...

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> FFmpegVideoFrames: ...

    @overload
    def __getitem__(self, arg: int, /) -> FFmpegImageFrames: ...

    @overload
    def __getitem__(self, arg: Sequence[int], /) -> FFmpegVideoFrames: ...

    def __repr__(self) -> str: ...

    def clone(self) -> FFmpegVideoFrames: ...

class ImagePackets:
    def __repr__(self) -> str: ...

    def clone(self) -> ImagePackets: ...

class InternalError(AssertionError):
    pass

class MMapAdaptor:
    def __init__(self) -> None: ...

class SourceAdaptor:
    def __init__(self) -> None: ...

class StreamingAudioDemuxer:
    pass

class StreamingVideoDecoder:
    pass

class StreamingVideoDemuxer:
    pass

class TracingSession:
    def init(self) -> None: ...

    def config(self, arg: str, /) -> None: ...

    def start(self, arg0: int, arg1: int, /) -> None: ...

    def stop(self) -> None: ...

class VideoPackets:
    @property
    def timestamp(self) -> tuple[float, float] | None: ...

    def __len__(self) -> int: ...

    def __repr__(self) -> str: ...

    def clone(self) -> VideoPackets: ...

@overload
def convert_frames(frames: FFmpegAudioFrames) -> CPUBuffer: ...

@overload
def convert_frames(frames: FFmpegVideoFrames) -> CPUBuffer: ...

@overload
def convert_frames(frames: FFmpegImageFrames) -> CPUBuffer: ...

@overload
def convert_frames(frames: Sequence[FFmpegAudioFrames]) -> CPUBuffer: ...

@overload
def convert_frames(frames: Sequence[FFmpegVideoFrames]) -> CPUBuffer: ...

@overload
def convert_frames(frames: Sequence[FFmpegImageFrames]) -> CPUBuffer: ...

@overload
def decode_image_nvjpeg(data: bytes, *, cuda_config: CUDAConfig, scale_width: int = -1, scale_height: int = -1, pix_fmt: str = 'rgb', _zero_clear: bool = False) -> CUDABuffer: ...

@overload
def decode_image_nvjpeg(data: Sequence[bytes], *, cuda_config: CUDAConfig, scale_width: int, scale_height: int, pix_fmt: str = 'rgb', _zero_clear: bool = False) -> CUDABuffer: ...

@overload
def decode_packets(packets: AudioPackets, *, decode_config: DecodeConfig | None = None, filter_desc: str = '') -> FFmpegAudioFrames: ...

@overload
def decode_packets(packets: VideoPackets, *, decode_config: DecodeConfig | None = None, filter_desc: str = '') -> FFmpegVideoFrames: ...

@overload
def decode_packets(packets: ImagePackets, *, decode_config: DecodeConfig | None = None, filter_desc: str = '') -> FFmpegImageFrames: ...

@overload
def decode_packets_nvdec(packets: VideoPackets, *, cuda_config: CUDAConfig, crop_left: int = 0, crop_top: int = 0, crop_right: int = 0, crop_bottom: int = 0, width: int = -1, height: int = -1, pix_fmt: str | None = 'rgba') -> CUDABuffer: ...

@overload
def decode_packets_nvdec(packets: ImagePackets, *, cuda_config: CUDAConfig, crop_left: int = 0, crop_top: int = 0, crop_right: int = 0, crop_bottom: int = 0, width: int = -1, height: int = -1, pix_fmt: str | None = 'rgba') -> CUDABuffer: ...

@overload
def decode_packets_nvdec(packets: Sequence[ImagePackets], *, cuda_config: CUDAConfig, crop_left: int = 0, crop_top: int = 0, crop_right: int = 0, crop_bottom: int = 0, width: int = -1, height: int = -1, pix_fmt: str | None = 'rgba', strict: bool = True) -> CUDABuffer: ...

@overload
def demux_audio(src: str, *, timestamp: tuple[float, float] | None = None, demux_config: DemuxConfig | None = None, _adaptor: SourceAdaptor | None = None) -> AudioPackets: ...

@overload
def demux_audio(data: bytes, *, timestamp: tuple[float, float] | None = None, demux_config: DemuxConfig | None = None, _zero_clear: bool = False) -> AudioPackets: ...

@overload
def demux_image(src: str, *, demux_config: DemuxConfig | None = None, _adaptor: SourceAdaptor | None = None) -> ImagePackets: ...

@overload
def demux_image(data: bytes, *, demux_config: DemuxConfig | None = None, _zero_clear: bool = False) -> ImagePackets: ...

@overload
def demux_video(src: str, *, timestamp: tuple[float, float] | None = None, demux_config: DemuxConfig | None = None, _adaptor: SourceAdaptor | None = None) -> VideoPackets: ...

@overload
def demux_video(data: bytes, *, timestamp: tuple[float, float] | None = None, demux_config: DemuxConfig | None = None, _zero_clear: bool = False) -> VideoPackets: ...

@overload
def encode_image(path: str, data: Annotated[ArrayLike, dict(dtype='uint8', device='cpu', order='C')], *, pix_fmt: str = 'rgb24', encode_config: EncodeConfig | None = None) -> None: ...

@overload
def encode_image(path: str, data: Annotated[ArrayLike, dict(dtype='uint8', device='cuda', order='C')], *, pix_fmt: str = 'rgb24', encode_config: EncodeConfig | None = None) -> None: ...

def get_ffmpeg_log_level() -> int: ...

def init_folly(arg0: str, arg1: Sequence[str], /) -> list[str]: ...

def init_tracing() -> TracingSession: ...

def is_cuda_available() -> bool: ...

def is_nvcodec_available() -> bool: ...

def register_avdevices() -> None: ...

def set_ffmpeg_log_level(arg: int, /) -> None: ...

@overload
def trace_counter(arg0: int, arg1: int, /) -> None: ...

@overload
def trace_counter(arg0: int, arg1: float, /) -> None: ...

def trace_event_begin(arg: str, /) -> None: ...

def trace_event_end() -> None: ...

@overload
def transfer_buffer(buffer: CPUBuffer, *, cuda_config: CUDAConfig) -> CUDABuffer: ...

@overload
def transfer_buffer(buffer: Annotated[ArrayLike, dict(device='cpu', order='C')], *, cuda_config: CUDAConfig) -> CUDABuffer: ...

def transfer_buffer_cpu(buffer: Annotated[ArrayLike, dict(device='cuda', order='C')]) -> CPUBuffer: ...
