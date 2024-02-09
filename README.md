# SPDL /spiːdl/

SPDL (Scalable and Performant Data Loading) is a library to provide fast
audio/video data loading for machine learning training.

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

* [FFmpeg](https://github.com/FFmpeg/FFmpeg) ([LGPL](https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv2.1)†)


* [PyBind11](https://github.com/pybind/pybind11) ([BSD-style](https://github.com/pybind/pybind11/blob/v2.11.1/LICENSE))

### Optional Dependencies

* [Perfetto](https://perfetto.dev/docs/instrumentation/tracing-sdk) ([Apache 2.0](https://github.com/google/perfetto/blob/v41.0/LICENSE))

† FFmpeg is dual-licensed software. One can choose LGPL or GPL. When building `libspdl`, pre-built FFmpeg library files are downloaded and linked against `libspdl`. These FFmpeg library files are compiled in a way that no GPL component is used and runtime search path is not hard-coded. Therefore, the resulting `libspdl` is not obliged to be GPL, and users can (need to) provide own FFmpeg library files.    
Users are free to dynamically link GPL or non-distributable version of FFmpeg libraries. However, note that linking a non-LGPL binary might change of the condition for redistribution of your application.
