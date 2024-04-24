# Installation

## Installing from source

The following command will build and install `spdl` Python package.

```
pip install . -v
```
Note: Make sure to use `-v` to see the log from the build process.

If you have CUDA GPUs, set the environment variable, `SPDL_USE_CUDA=1` to enable
CUDA integration.

If you want to use [NVIDIA VIDEO CODEC](https://developer.nvidia.com/video-codec-sdk) integration, set `SPDL_USE_NVCODEC=1`. SPDL will build with a custom stub library.

To enable Perfetto tracing, set `SPDL_USE_TRACING=1`.

* Development installation

```
pip install -e . -v
```

The build process takes multiple stages to resolve the dependencies.
First, it fetches, builds and installs Folly dependencies to build directory.
Second, it builds SPDL, Folly and other packates that SPDL directly depends on.

Once Folly dependencies are built, setting the environment variable
`SKIP_FOLLY_DEPS=1` and the build will skip the Folly dependency in subsequent builds.

## Requirements

* Supported OS: Linux, macOS

* Build requirements
    - C++20 compiler (Tested on GCC 11 and Clang 15)
    - CMake, Ninja
    - (Optional) CUDA Toolkit
    

* Runtime Requirements
    - Python 3.10+

* Dependencies
    - NumPy
    - (Optional) PyTorch
    - (Optional) Numba
