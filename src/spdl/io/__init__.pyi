# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Type stub file for spdl.io module."""

# Python modules
from spdl.io._array import (
    load_npy as load_npy,
    load_npz as load_npz,
    NpzFile as NpzFile,
)
from spdl.io._composite import (
    load_audio as load_audio,
    load_image as load_image,
    load_image_batch as load_image_batch,
    load_image_batch_nvjpeg as load_image_batch_nvjpeg,
    load_video as load_video,
    sample_decode_video as sample_decode_video,
    save_image as save_image,
    streaming_load_video_nvdec as streaming_load_video_nvdec,
)
from spdl.io._config import (
    audio_encode_config as audio_encode_config,
    cpu_storage as cpu_storage,
    cuda_config as cuda_config,
    decode_config as decode_config,
    demux_config as demux_config,
    video_encode_config as video_encode_config,
)
from spdl.io._convert import (
    ArrayInterface as ArrayInterface,
    CUDAArrayInterface as CUDAArrayInterface,
    to_jax as to_jax,
    to_numba as to_numba,
    to_numpy as to_numpy,
    to_torch as to_torch,
)
from spdl.io._core import (
    apply_bsf as apply_bsf,
    BSF as BSF,
    convert_array as convert_array,
    convert_frames as convert_frames,
    create_reference_audio_frame as create_reference_audio_frame,
    create_reference_video_frame as create_reference_video_frame,
    decode_image_nvjpeg as decode_image_nvjpeg,
    decode_packets as decode_packets,
    decode_packets_nvdec as decode_packets_nvdec,
    Decoder as Decoder,
    demux_audio as demux_audio,
    demux_image as demux_image,
    demux_video as demux_video,
    Demuxer as Demuxer,
    encode_image as encode_image,
    Muxer as Muxer,
    nv12_to_bgr as nv12_to_bgr,
    nv12_to_rgb as nv12_to_rgb,
    nvdec_decoder as nvdec_decoder,
    NvDecDecoder as NvDecDecoder,
    transfer_buffer as transfer_buffer,
    transfer_buffer_cpu as transfer_buffer_cpu,
)
from spdl.io._preprocessing import (
    get_abuffer_desc as get_abuffer_desc,
    get_audio_filter_desc as get_audio_filter_desc,
    get_buffer_desc as get_buffer_desc,
    get_filter_desc as get_filter_desc,
    get_video_filter_desc as get_video_filter_desc,
)
from spdl.io._tar import iter_tarfile as iter_tarfile
from spdl.io._transfer import transfer_tensor as transfer_tensor
from spdl.io._wav import load_wav as load_wav, parse_wav as parse_wav

# C++ extension classes and functions from _libspdl
from spdl.io.lib._libspdl import (
    AudioCodec as AudioCodec,
    AudioDecoder as AudioDecoder,
    AudioEncodeConfig as AudioEncodeConfig,
    AudioEncoder as AudioEncoder,
    AudioFrames as AudioFrames,
    AudioPackets as AudioPackets,
    CPUBuffer as CPUBuffer,
    CPUStorage as CPUStorage,
    DecodeConfig as DecodeConfig,
    DemuxConfig as DemuxConfig,
    FilterGraph as FilterGraph,
    ImageCodec as ImageCodec,
    ImageFrames as ImageFrames,
    ImagePackets as ImagePackets,
    VideoCodec as VideoCodec,
    VideoDecoder as VideoDecoder,
    VideoEncodeConfig as VideoEncodeConfig,
    VideoEncoder as VideoEncoder,
    VideoFrames as VideoFrames,
    VideoPackets as VideoPackets,
)

# C++ extension classes and functions from _libspdl_cuda
from spdl.io.lib._libspdl_cuda import (
    CUDABuffer as CUDABuffer,
    CUDAConfig as CUDAConfig,
    NvDecDecoder as NvDecDecoder,
)

# C++ extension classes and functions from _wav
from spdl.io.lib._wav import (
    WAVHeader as WAVHeader,
)

__all__ = [
    # From _array
    "load_npy",
    "load_npz",
    "NpzFile",
    # From _composite
    "load_audio",
    "load_image",
    "load_image_batch",
    "load_image_batch_nvjpeg",
    "load_video",
    "sample_decode_video",
    "save_image",
    "streaming_load_video_nvdec",
    # From _config
    "audio_encode_config",
    "cpu_storage",
    "cuda_config",
    "decode_config",
    "demux_config",
    "video_encode_config",
    # From _convert
    "ArrayInterface",
    "CUDAArrayInterface",
    "to_jax",
    "to_numba",
    "to_numpy",
    "to_torch",
    # From _core
    "apply_bsf",
    "BSF",
    "convert_array",
    "convert_frames",
    "create_reference_audio_frame",
    "create_reference_video_frame",
    "decode_image_nvjpeg",
    "decode_packets",
    "decode_packets_nvdec",
    "Decoder",
    "demux_audio",
    "demux_image",
    "demux_video",
    "Demuxer",
    "encode_image",
    "Muxer",
    "nv12_to_bgr",
    "nv12_to_rgb",
    "nvdec_decoder",
    "NvDecDecoder",
    "transfer_buffer",
    "transfer_buffer_cpu",
    # From _preprocessing
    "FilterGraph",
    "get_abuffer_desc",
    "get_audio_filter_desc",
    "get_buffer_desc",
    "get_filter_desc",
    "get_video_filter_desc",
    # From _tar
    "iter_tarfile",
    # From _transfer
    "transfer_tensor",
    # From _wav
    "load_wav",
    "parse_wav",
    "WAVHeader",
    # From lib._libspdl
    "AudioCodec",
    "AudioDecoder",
    "AudioEncodeConfig",
    "AudioEncoder",
    "AudioFrames",
    "AudioPackets",
    "CPUBuffer",
    "CPUStorage",
    "DecodeConfig",
    "DemuxConfig",
    "ImageCodec",
    "ImageFrames",
    "ImagePackets",
    "VideoCodec",
    "VideoDecoder",
    "VideoEncodeConfig",
    "VideoEncoder",
    "VideoFrames",
    "VideoPackets",
    # From lib._libspdl_cuda
    "CUDABuffer",
    "CUDAConfig",
]
