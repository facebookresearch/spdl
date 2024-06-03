# SPDL /spiːdl/

SPDL (Scalable and Performant Data Loading) is a library to provide fast
audio/video/image data loading for machine learning training.

## Examples

Please checkout prototypes.

- [ImageNet evaluation](./src/prototypes/imagenet_classification.py)

- [Image dataloading](./src/prototypes/image_dataloading.py)
Loads the training set (1281167 images) in 40 seconds.

- [Video dataloading](./src/prototypes/video_dataloading.py)
Loads the Kinetics 400 trainig dataset in 15 mins.

## Installation

Please refer to [the documentation](./docs/installation.md).

## Documentation

Documentations are found in the [docs](./docs) directory.

Please use mkdocs to build and browse the documentation after installing SPDL.

```
mkdoc serve
```

## Dependencies

### Third party libraries

The libspdl uses the following third party libraries.

* [Folly](https://github.com/facebook/folly) ([Apache 2](https://github.com/facebook/folly/blob/main/LICENSE)) and its dependencies
   * [Boost](https://github.com/boostorg/boost/) ([Boost Software License 1.0](https://github.com/boostorg/boost/blob/boost-1.84.0/LICENSE_1_0.txt))
   * [Double Conversion](https://github.com/google/double-conversion) ([BSD-3](https://github.com/google/double-conversion/blob/v3.3.0/LICENSE))
   * [{fmt}](https://github.com/fmtlib/fmt) ([MIT](https://github.com/fmtlib/fmt/blob/10.1.1/LICENSE.rst))
   * [gflags](https://github.com/gflags/gflags) ([BSD-3](https://github.com/gflags/gflags/blob/v2.2.0/COPYING.txt))
   * [glog](https://github.com/google/glog) ([BSD-3](https://github.com/google/glog/blob/v0.5.0/COPYING))
   * [Libevent](https://github.com/libevent/libevent) ([BSD-3](https://github.com/mthrok/libevent/blob/release-2.1.12-stable-patch/LICENSE))

* [nanobind](https://github.com/wjakob/nanobind) ([BSD-3](https://github.com/wjakob/nanobind/blob/v1.9.0/LICENSE)) and its dependency
   * [robin-map](https://github.com/Tessil/robin-map/) ([MIT](https://github.com/Tessil/robin-map/blob/v1.3.0/LICENSE))

* [FFmpeg](https://github.com/FFmpeg/FFmpeg) ([LGPL](https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv2.1)†)

### Optional Dependencies

* [Perfetto](https://perfetto.dev/docs/instrumentation/tracing-sdk) ([Apache 2.0](https://github.com/google/perfetto/blob/v41.0/LICENSE))

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)†† ([CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html))    
When building with CUDA integration, CUDA header files must be installed.
    * [nvJPEG](https://docs.nvidia.com/cuda/nvjpeg/index.html)†† (Also covered by CUDA Toolkit EULA)

* [Video Codec SDK](https://gitlab.com/nvidia/video/video-codec-sdk)†† header files    
The header files of video codec SDK (`nvcuvid.h` and `cuviddec.h`), which are distribtued under MIT license, is used when compiling SPDL with hardware video decoder enabled.


† FFmpeg is dual-licensed software. One can choose LGPL or GPL. When building `libspdl`, pre-built LGPL version of FFmpeg library files are downloaded and linked against `libspdl`. These FFmpeg library files are compiled in a way that no GPL component is used and runtime search path is not hard-coded. Therefore, the resulting `libspdl` is not obliged to be GPL, and users can (need to) provide own FFmpeg library files.

Users are free to dynamically link GPL or non-distributable version of FFmpeg libraries. However, note that linking a non-LGPL binary might change of the condition for redistribution of your application.

†† This software contains source code provided by NVIDIA Corporation.
