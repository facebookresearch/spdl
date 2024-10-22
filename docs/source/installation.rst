Installation
============

From source
-----------

The following command will build and install `spdl` Python package.

.. note::

   Make sure to use `-v` to see the log from the actual build process.

**Regular installation**

.. code-block::

   pip install . -v

**Development installation**

.. code-block::

   pip install -e . -v

The build process first downloads/builds/installs some third-party
dependencies, then it builds SPDL and its binding code.

Build can be customized through the environment variables;

- ``SPDL_USE_CUDA=1``: Enable CUDA integration, such as background data transfer.
- ``SPDL_USE_NVCODEC=1``: Enable
  `NVIDIA VIDEO CODEC <https://developer.nvidia.com/video-codec-sdk>`_
  integration, i.e. GPU video decoder and direct CUDA memory placement
- ``SPDL_USE_TRACING=1``: Enable `Perfetto <https://perfetto.dev/>`_
  integration for performance profiling.
- ``SPDL_USE_FFMPEG_VERSION``: Specify the version of FFmpeg you want to use
  to reduce the build time. By default, SPDL compiles against FFmpeg 4, 5, 6 and 7,
  and pick available one at run time.
- ``SPDL_RELASE_GIL=0``: I/O module does not release GIL. This is intended to be used for
  experimenting with free-threaded Python (3.13+).

See `setup.py <https://github.com/facebookresearch/spdl/blob/main/setup.py>`_ for the up-to-date available options.

Trouble Shooting
----------------

If you hit an error like the following, ``--no-build-isolation`` can help.

.. code-block:: text

   Installing collected packages: ninja, wheel, setuptools, cmake
     Creating /tmp/pip-build-env-1ac3bsc_/overlay/bin
     changing mode of /tmp/pip-build-env-1ac3bsc_/overlay/bin/ninja to 775
     changing mode of /tmp/pip-build-env-1ac3bsc_/overlay/bin/wheel to 775
     changing mode of /tmp/pip-build-env-1ac3bsc_/overlay/bin/cmake to 775
     changing mode of /tmp/pip-build-env-1ac3bsc_/overlay/bin/cpack to 775
     changing mode of /tmp/pip-build-env-1ac3bsc_/overlay/bin/ctest to 775
   Successfully installed cmake-3.30.5 ninja-1.11.1.1 setuptools-75.2.0 wheel-0.44.0

   ...

   CMake Error at CMakeLists.txt:3 (project):
     Running

      '/tmp/pip-build-env-xic7ygls/overlay/bin/ninja' '--version'

     failed with:

      no such file or directory

This happens because, somehow ``cmake`` command is not pointing the ``ninja`` installed
by ``pip``.

One workaround is to use ``--no-build-isolation`` option.
e.g. ``TMPDIR="${PWD}/build" pip install . -v  --no-build-isolation``


Requirements
------------

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

Dependencies
------------

The libspdl uses the following third party libraries, which are fetched and built automatically during the build process.

* `{fmt} <https://github.com/fmtlib/fmt>`_ (`MIT <https://github.com/fmtlib/fmt/blob/10.1.1/LICENSE.rst>`_)
* `gflags <https://github.com/gflags/gflags>`_ (`BSD-3 <https://github.com/gflags/gflags/blob/v2.2.0/COPYING.txt>`_)
* `glog <https://github.com/google/glog>`_ (`BSD-3 <https://github.com/google/glog/blob/v0.5.0/COPYING>`_)
* `nanobind <https://github.com/wjakob/nanobind>`_ (`BSD-3 <https://github.com/wjakob/nanobind/blob/v2.0.0/LICENSE>`_) and its dependency `robin-map <https://github.com/Tessil/robin-map/>`_ (`MIT <https://github.com/Tessil/robin-map/blob/v1.3.0/LICENSE>`_)
* `FFmpeg <https://github.com/FFmpeg/FFmpeg>`_ (`LGPL <https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv2.1>`_ †)

.. note::

   **†** FFmpeg is a dual-licensed software. One can choose LGPL or GPL.

   When building ``libspdl``, pre-built LGPL version of FFmpeg library files are
   downloaded and linked against ``libspdl``.
   These FFmpeg library files are compiled in a way that no GPL component is used
   and runtime search path is not hard-coded.
   Therefore, the resulting ``libspdl`` is not obliged to be GPL, and
   users can (need to) provide own FFmpeg library files.

   Users are free to dynamically link GPL or non-distributable version of
   FFmpeg libraries. However, please note that linking a non-LGPL binary might
   change the condition for redistribution of your application.


Optional Dependencies
---------------------

* `Perfetto <https://perfetto.dev/docs/instrumentation/tracing-sdk>`_ (`Apache 2.0 <https://github.com/google/perfetto/blob/v41.0/LICENSE>`_)
* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ †† (`CUDA Toolkit EULA <https://docs.nvidia.com/cuda/eula/index.html>`_) and the following family of libraries covered by the same EULA    
    * `nvJPEG <https://docs.nvidia.com/cuda/nvjpeg/index.html>`_
    * `NPP <https://developer.nvidia.com/npp>`_
* The header files of `Video Codec SDK <https://gitlab.com/nvidia/video/video-codec-sdk>`_ ††

  The header files of video codec SDK (``nvcuvid.h`` and ``cuviddec.h``),
  which are distributed under MIT license, is used when compiling SPDL with
  hardware video decoder enabled.

.. note::

   **††** This software contains source code provided by NVIDIA Corporation.

Building with Free-Threaded Python
----------------------------------

To build SPDL with Free-Threaded Python, the following manual changes are required.
We intend to incorporate these changes in build process, once Python 3.13 and
FT-aware nanobind is released.

1. Add ``FREE_THREADED`` to ``nanobind_add_module``. Please refer to `the doc <https://nanobind.readthedocs.io/en/latest/free_threaded.html>`_.
