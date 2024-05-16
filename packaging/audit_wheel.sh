#!/usr/bin/env bash

auditwheel show dist/spdl*.whl
auditwheel repair                \
    dist/spdl*.whl               \
    --plat manylinux_2_27_x86_64 \
    --exclude libavcodec.so.58   \
    --exclude libavcodec.so.59   \
    --exclude libavcodec.so.60   \
    --exclude libavcodec.so.61   \
    --exclude libavfilter.so.7   \
    --exclude libavfilter.so.8   \
    --exclude libavfilter.so.9   \
    --exclude libavfilter.so.10  \
    --exclude libavdevice.so.58  \
    --exclude libavdevice.so.59  \
    --exclude libavdevice.so.60  \
    --exclude libavdevice.so.61  \
    --exclude libavformat.so.58  \
    --exclude libavformat.so.59  \
    --exclude libavformat.so.60  \
    --exclude libavformat.so.61  \
    --exclude libavutil.so.56    \
    --exclude libavutil.so.57    \
    --exclude libavutil.so.58    \
    --exclude libavutil.so.59    \
    --exclude libnvcuvid.so      \
    --exclude libcudart.so.11.0  \
    --exclude libcuda.so.1
