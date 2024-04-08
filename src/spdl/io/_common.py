from spdl.lib import _libspdl


def _get_demux_func(media_type, src):
    match media_type:
        case "audio":
            if isinstance(src, bytes):
                name = "async_demux_audio_bytes"
            else:
                name = "async_demux_audio"
        case "video":
            if isinstance(src, bytes):
                name = "async_demux_video_bytes"
            else:
                name = "async_demux_video"
        case "image":
            if isinstance(src, bytes):
                name = "async_demux_image_bytes"
            else:
                name = "async_demux_image"
        case _:
            raise ValueError(f"Unexpected media type: {media_type}.")
    return getattr(_libspdl, name)


def _get_decoding_func(packets):
    match t := type(packets):
        case _libspdl.AudioPackets:
            name = "async_decode_audio"
        case _libspdl.VideoPackets:
            name = "async_decode_video"
        case _libspdl.ImagePackets:
            name = "async_decode_image"
        case _:
            raise TypeError(f"Unexpected type: {t}.")
    return getattr(_libspdl, name)


def _get_nvdec_decoding_func(packets):
    match t := type(packets):
        case _libspdl.VideoPackets:
            name = "async_decode_video_nvdec"
        case _libspdl.ImagePackets:
            name = "async_decode_image_nvdec"
        case _:
            raise TypeError(f"Unexpected type: {t}.")
    return getattr(_libspdl, name)


def _get_cpu_conversion_func(frames):
    match t := type(frames):
        case _libspdl.FFmpegAudioFrames:
            name = "async_convert_audio_cpu"
        case _libspdl.FFmpegVideoFrames:
            name = "async_convert_video_cpu"
        case _libspdl.FFmpegImageFrames:
            name = "async_convert_image_cpu"
        case _:
            if not isinstance(frames, list):
                raise TypeError(f"Unexpected type: {t}.")
            if any(not isinstance(f, _libspdl.FFmpegImageFrames) for f in frames):
                raise TypeError(
                    f"Unexpected type: {t}. When the container type is list, all frames must be of type FFmpegImageFrames."
                )
            name = "async_convert_batch_image_cpu"
    return getattr(_libspdl, name)


def _get_conversion_name(frames):
    match t := type(frames):
        case _libspdl.FFmpegAudioFrames:
            name = "async_convert_audio"
        case _libspdl.FFmpegVideoFrames:
            name = "async_convert_video"
        case _libspdl.FFmpegImageFrames:
            name = "async_convert_image"
        case _libspdl.NvDecVideoFrames:
            name = "async_convert_video_nvdec"
        case _libspdl.NvDecImageFrames:
            name = "async_convert_image_nvdec"
        case _:
            if not isinstance(frames, list):
                raise TypeError(f"Unexpected type: {t}.")
            if all(isinstance(f, _libspdl.FFmpegImageFrames) for f in frames):
                name = "async_convert_batch_image"
            elif all(isinstance(f, _libspdl.NvDecImageFrames) for f in frames):
                name = "async_convert_batch_image_nvdec"
            else:
                raise TypeError(
                    f"Unexpected type: {t}. When the container type is list, all frames must be either FFmpegImageFrames or NvDecImageFrames."
                )
    return getattr(_libspdl, name)