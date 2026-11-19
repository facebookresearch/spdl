#!/usr/bin/env bash

# This script builds FFmpeg libraries without any functional features.
#
# IMPORTANT:
# The resulting library files have to be LGPL version of FFmpeg libraries.
# - Do not enable `--enable-nonfree` and `--enable-gpl`.
# - Do not enable third party library integrations like x264.
#
# This script is not meant to build useful FFmpeg libraries, but to build
# a skeleton of FFmpeg libraries that are use only during the build process of
# torchaudio.
#
# The resulting FFmpeg libraries should not be shipped either.

set -eux

acquire_ffmpeg_source() {
    local ffmpeg_version="$1"
    local archive
    if [[ "${ffmpeg_version}" == master ]]; then
        archive="https://github.com/FFmpeg/FFmpeg/archive/master.tar.gz"
    else
        archive="https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${ffmpeg_version}.tar.gz"
    fi

    # NOTE:
    # When changing the version of FFmpeg, update the README so that the link to the source points
    # the same version.
    # Reject HTTP errors and let curl retry transient failures, with a bounded retry budget.
    curl -fLsS --retry 3 --retry-max-time 60 -o ffmpeg.tar.gz "${archive}"
    tar -xf ffmpeg.tar.gz --strip-components 1
}

configure_ffmpeg() {
    local prefix="$1"
    # Keep the array nonempty even without platform-specific options: Bash 3.2
    # treats an empty array expansion as unset under `set -u`.
    local configure_args=(
        "--prefix=${prefix}"
        --disable-all
        --disable-everything
        --disable-programs
        --disable-doc
        --disable-debug
        --disable-autodetect
        --disable-x86asm
        --disable-iconv
        --disable-encoders
        --disable-decoders
        --disable-hwaccels
        --disable-muxers
        --disable-demuxers
        --disable-parsers
        --disable-bsfs
        --disable-protocols
        --disable-devices
        --disable-filters
        --disable-asm
        --disable-static
        --enable-shared
        --enable-rpath
        --enable-pic
        --enable-avcodec
        --enable-avdevice
        --enable-avfilter
        --enable-avformat
        --enable-avutil
    )
    if [[ "$OSTYPE" == msys ]]; then
        configure_args+=(--toolchain=msvc)
    fi
    ./configure "${configure_args[@]}"
}

# Scope the working directory and traps to this build, not the calling shell
# or the later macOS library repair. All exit paths clean up the temporary source.
build_ffmpeg() (
    local prefix="$1"
    local ffmpeg_version="$2"
    local build_dir
    local status
    build_dir=$(mktemp -d -t ffmpeg-build.XXXXXXXXXX)
    # Cleanup must not replace a failed build's status, even if rm itself fails.
    trap 'status=$?; rm -rf -- "${build_dir}" || :; exit "${status}"' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    cd "${build_dir}"
    acquire_ffmpeg_source "${ffmpeg_version}"
    configure_ffmpeg "${prefix}"
    make -j install
    ls "${prefix}"/*
)

show_macos_library_paths() {
    local prefix="$1"
    local otool="$2"
    local lib="$3"
    "${otool}" -l "${prefix}/lib/${lib}.dylib" | grep -F -B2 -- "${prefix}"
}

# Each dependency is supplied in the same order as the original linker rewrites.
repair_macos_library() {
    local prefix="$1"
    local otool="$2"
    local install_name_tool="$3"
    local lib="$4"
    shift 4
    local dependency
    local rewrite_args=()
    for dependency in "$@"; do
        rewrite_args+=(
            -change "${prefix}/lib/${dependency}.dylib" "@rpath/${dependency}.dylib"
        )
    done
    rewrite_args+=(
        -delete_rpath "${prefix}/lib"
        -id "@rpath/${lib}.dylib"
        "${prefix}/lib/${lib}.dylib"
    )
    "${install_name_tool}" "${rewrite_args[@]}"
    show_macos_library_paths "${prefix}" "${otool}" "${lib}"
}

repair_macos_libraries() {
    local prefix="$1"
    local ffmpeg_version="$2"
    local major_ver="${ffmpeg_version:0:1}"
    local avutil avcodec avformat avdevice avfilter
    case "${ffmpeg_version}" in
        4*)
            avutil=libavutil.56
            avcodec=libavcodec.58
            avformat=libavformat.58
            avdevice=libavdevice.58
            avfilter=libavfilter.7
            ;;
        5*)
            avutil=libavutil.57
            avcodec=libavcodec.59
            avformat=libavformat.59
            avdevice=libavdevice.59
            avfilter=libavfilter.8
            ;;
        6*)
            avutil=libavutil.58
            avcodec=libavcodec.60
            avformat=libavformat.60
            avdevice=libavdevice.60
            avfilter=libavfilter.9
            ;;
        7*)
            avutil=libavutil.59
            avcodec=libavcodec.61
            avformat=libavformat.61
            avdevice=libavdevice.61
            avfilter=libavfilter.10
            ;;
        8*)
            avutil=libavutil.60
            avcodec=libavcodec.62
            avformat=libavformat.62
            avdevice=libavdevice.62
            avfilter=libavfilter.11
            ;;
        9*|master)
            avutil=libavutil.61
            avcodec=libavcodec.63
            avformat=libavformat.63
            avdevice=libavdevice.63
            avfilter=libavfilter.12
            ;;
        *)
            printf 'Error: unexpected FFmpeg major version: %s\n' "${major_ver}"
            return 1
            ;;
    esac

    # Prefer system tools to miniconda versions, which can give inconsistent results.
    local otool="/usr/bin/otool"
    local install_name_tool="/usr/bin/install_name_tool"
    if [[ ! -e "${otool}" ]]; then
        otool="$(which otool)"
    fi
    if [[ ! -e "${install_name_tool}" ]]; then
        install_name_tool="$(which install_name_tool)"
    fi

    local lib
    for lib in "${avcodec}" "${avdevice}" "${avfilter}" "${avformat}" "${avutil}"; do
        show_macos_library_paths "${prefix}" "${otool}" "${lib}"
    done

    repair_macos_library "${prefix}" "${otool}" "${install_name_tool}" "${avcodec}" \
        "${avutil}"
    repair_macos_library "${prefix}" "${otool}" "${install_name_tool}" "${avdevice}" \
        "${avformat}" "${avcodec}" "${avutil}"
    repair_macos_library "${prefix}" "${otool}" "${install_name_tool}" "${avfilter}" \
        "${avutil}"
    repair_macos_library "${prefix}" "${otool}" "${install_name_tool}" "${avformat}" \
        "${avcodec}" "${avutil}"
    repair_macos_library "${prefix}" "${otool}" "${install_name_tool}" "${avutil}"
}

prefix="${FFMPEG_ROOT}"
ffmpeg_version="${FFMPEG_VERSION:-4.1.8}"
build_ffmpeg "${prefix}" "${ffmpeg_version}"

# macOS: Fix rpath so that the libraries are searched dynamically in user environment.
# In Linux, this is handled by the configure option --enable-rpath.
if [[ "$(uname)" == Darwin ]]; then
    repair_macos_libraries "${prefix}" "${ffmpeg_version}"
fi
