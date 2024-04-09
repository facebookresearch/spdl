#pragma once

#include <libspdl/core/types.h>

#include <cuviddec.h>
#include <nvcuvid.h>

#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace spdl::core::detail {

cudaVideoCodec covert_codec_id(AVCodecID);

void check_support(CUVIDEOFORMAT* fmt, CUVIDDECODECAPS caps);

bool is_compatible(const CUVIDEOFORMAT* fmt, const CUVIDDECODECAPS& caps);

void reconfigure_decoder(
    CUvideodecoder decoder,
    const CUVIDDECODECREATEINFO& create_info);

CUvideodecoder get_decoder(CUVIDDECODECREATEINFO* param);

CUVIDDECODECREATEINFO get_create_info(
    CUvideoctxlock lock,
    CUVIDEOFORMAT* video_fmt,
    cudaVideoSurfaceFormat surface_fmt,
    unsigned int max_width,
    unsigned int max_height,
    const CropArea& crop,
    int target_width,
    int target_height);

cudaVideoSurfaceFormat get_output_sufrace_format(
    CUVIDEOFORMAT* video_fmt,
    CUVIDDECODECAPS* decode_caps);

CUVIDDECODECAPS check_capacity(
    CUVIDEOFORMAT* video_fmt,
    std::vector<CUVIDDECODECAPS>& cache);

const char* get_desc(cuvidDecodeStatus);

const char* get_surface_format_name(cudaVideoSurfaceFormat surface_fmt);
const char* get_video_format_name(unsigned char video_fmt);
const char* get_chroma_name(cudaVideoChromaFormat chroma);
const char* get_codec_name(cudaVideoCodec codec);

std::string print(const CUVIDEOFORMAT*);
std::string print(const CUVIDDECODECAPS*);
std::string print(const CUVIDPICPARAMS*);
std::string print(const CUVIDPARSERDISPINFO*);
std::string print(const CUVIDSEIMESSAGEINFO*);
std::string print(const CUVIDDECODECREATEINFO*);

std::string get_diff(
    const CUVIDDECODECREATEINFO& i1,
    const CUVIDDECODECREATEINFO& i2);

} // namespace spdl::core::detail
