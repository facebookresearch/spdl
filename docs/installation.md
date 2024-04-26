# Installation

## Installing from source

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

The build process takes multiple stages to resolve the dependencies.
First, it fetches, builds and installs Folly dependencies in the build directory.
Second, it builds SPDL, Folly and other packages that SPDL directly depends on.

Once Folly dependencies are built, set the environment variable
`SKIP_FOLLY_DEPS=1` and the build will skip the Folly dependency in subsequent builds.

Build can be customized through the environment variables;

- `SPDL_USE_CUDA=1` to enable CUDA integration, such as background data transfer.
- `SPDL_USE_NVCODEC=1` to enable [NVIDIA VIDEO CODEC](https://developer.nvidia.com/video-codec-sdk)
   integration, i.e. GPU video decoder and direct CUDA memory placement
- `SPDL_USE_TRACING=1` to [Perfetto](https://perfetto.dev/) integration for performance profiling.
- `SPDL_USE_FFMPEG_VERSION` to specify the version of FFmpeg you want to use to reduce the build time.
   By default, SPDL compiles against FFmpeg 4, 5, 6 and 7, and pick available one at run time.

See [setup.py](https://github.com/mthrok/spdl/blob/main/setup.py) for the available options.

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
