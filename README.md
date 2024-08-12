# SPDL /spiːdl/

SPDL (Scalable and Performant Data Loading) is a library to provide fast
audio/video/image data loading for machine learning training.

## Examples

Please checkout prototypes.

- [ImageNet evaluation](./examples/imagenet_classification.py)

- [Image dataloading](./examples/image_dataloading.py)
Loads the training set (1281167 images) in 40 seconds.

- [Video dataloading](./src/prototypes/video_dataloading.py)
Loads the Kinetics 400 trainig dataset in 15 mins.

## Installation

### Installing from source

The following command will build and install `spdl` Python package.

_Note_ Make sure to use `-v` to see the log from the build process.

_Regular installation_

```
pip install . -v
```

_Development installation_

```
pip install -e . -v
```

The build process first downloads/builds/installs some third-party dependencies, then it builds SPDL and its binding code.

Once dependencies are built, set the environment variable
`SKIP_1ST_DEPS=1` and the build will skip the Folly dependency in subsequent builds.

Build can be customized through the environment variables;

- `SPDL_USE_CUDA=1` to enable CUDA integration, such as background data transfer.
- `SPDL_USE_NVCODEC=1` to enable [NVIDIA VIDEO CODEC](https://developer.nvidia.com/video-codec-sdk) integration, i.e. GPU video decoder and direct CUDA memory placement
- `SPDL_USE_TRACING=1` to [Perfetto](https://perfetto.dev/) integration for performance profiling.
- `SPDL_USE_FFMPEG_VERSION` to specify the version of FFmpeg you want to use to reduce the build time.
   By default, SPDL compiles against FFmpeg 4, 5, 6 and 7, and pick available one at run time.

See [setup.py](https://github.com/facebookresearch/spdl/blob/main/setup.py) for the up-to-date available options.

## Requirements

* Supported OS: Linux, macOS

* Build requirements
    - C++20 compiler (Tested on GCC 11 and Clang 15)
    - CMake, Ninja
    - (Optional) CUDA Toolkit

* Runtime Requirements and dependencies
    - Python 3.10+
    - NumPy
    - (Optional) PyTorch
    - (Optional) Numba
    - (Optional) CUDA runtime

## Dependencies

### Third party libraries

The libspdl uses the following third party libraries.

* [{fmt}](https://github.com/fmtlib/fmt) ([MIT](https://github.com/fmtlib/fmt/blob/10.1.1/LICENSE.rst))
* [gflags](https://github.com/gflags/gflags) ([BSD-3](https://github.com/gflags/gflags/blob/v2.2.0/COPYING.txt))
* [glog](https://github.com/google/glog) ([BSD-3](https://github.com/google/glog/blob/v0.5.0/COPYING))
* [nanobind](https://github.com/wjakob/nanobind) ([BSD-3](https://github.com/wjakob/nanobind/blob/v2.0.0/LICENSE)) and its dependency
   * [robin-map](https://github.com/Tessil/robin-map/) ([MIT](https://github.com/Tessil/robin-map/blob/v1.3.0/LICENSE))
* [FFmpeg](https://github.com/FFmpeg/FFmpeg) ([LGPL](https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv2.1)†)

> † FFmpeg is dual-licensed software. One can choose LGPL or GPL. When building `libspdl`, pre-built LGPL version of FFmpeg library files are downloaded and linked against `libspdl`. These FFmpeg library files are compiled in a way that no GPL component is used and runtime search path is not hard-coded. Therefore, the resulting `libspdl` is not obliged to be GPL, and users can (need to) provide own FFmpeg library files.
>
> Users are free to dynamically link GPL or non-distributable version of FFmpeg libraries. However, note that linking a non-LGPL binary might change of the condition for redistribution of your application.


### Optional Dependencies

* [Perfetto](https://perfetto.dev/docs/instrumentation/tracing-sdk) ([Apache 2.0](https://github.com/google/perfetto/blob/v41.0/LICENSE))
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)†† ([CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)) and the following family of libraries covered by the same EULA    
    * [nvJPEG](https://docs.nvidia.com/cuda/nvjpeg/index.html)
    * [NPP](https://developer.nvidia.com/npp)
* [Video Codec SDK](https://gitlab.com/nvidia/video/video-codec-sdk)†† header files    
The header files of video codec SDK (`nvcuvid.h` and `cuviddec.h`), which are distribtued under MIT license, is used when compiling SPDL with hardware video decoder enabled.

> †† This software contains source code provided by NVIDIA Corporation.
