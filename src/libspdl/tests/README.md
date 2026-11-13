# libspdl C++ unit tests

The tests in this directory also run outside Buck using CMake and CTest. The
native test build is opt-in (`SPDL_BUILD_TESTS=ON`); ordinary SPDL library and
wheel builds do not build GoogleTest, generate media, or register tests.

## Local setup

From the repository root, create a separate Conda environment:

```bash
conda env create --prefix "$PWD/build/libspdl-tests/env" \
  --file src/libspdl/tests/environment.yml
conda activate "$PWD/build/libspdl-tests/env"
```

A C++20 compiler is required. On macOS, install the Xcode Command Line Tools.
An isolated Conda compiler can also be used, for example:

```bash
conda install -c conda-forge clangxx_osx-arm64=20
conda deactivate
conda activate "$PWD/build/libspdl-tests/env"
```

Build and run all core C++ tests against one FFmpeg major:

```bash
bash tools/build_cpp_tests.sh 8
```

Use a matching full FFmpeg runtime for every tested major. For FFmpeg 9, create
a separate environment and build tree rather than reusing the FFmpeg 8 runtime:

```bash
conda env create --prefix "$PWD/build/libspdl-tests-9/env" \
  --file src/libspdl/tests/environment.yml
conda activate "$PWD/build/libspdl-tests-9/env"
conda install -c conda-forge 'ffmpeg=9.*=gpl_*'
bash tools/build_cpp_tests.sh 9 "$PWD/build/libspdl-tests-9"
```

The script reuses SPDL's pinned source-built dependency stage. Native tests
link directly to full FFmpeg headers and libraries from
`SPDL_TEST_FFMPEG_ROOT` (default: the active Conda prefix). CMake derives and
validates the selected FFmpeg ABI from those headers. Ordinary wheel builds
continue using the existing multi-version link-time skeletons; this test setup
does not change that distribution path. The script rejects a mismatched FFmpeg
CLI major. It disables Python
bindings and unnecessary C++ module
scanning. The native-test dependency stage uses shared glog so the test's
`LogSink` sees library warnings; default library/wheel builds keep static glog.
`CMAKE_BUILD_PARALLEL_LEVEL` controls build and test concurrency
(default: 2). A second argument selects an alternative build directory; keep
compiler/toolchain changes in a fresh directory.

CMake downloads dependencies while configuring. Test execution itself needs no
network. Synthetic AAC, JPEG and H.264/MP4 fixtures are generated with the
Conda FFmpeg CLI at build time, not downloaded. The helper selects a matching
Conda runtime for each tested major; the generated media formats themselves are
independent of the library ABI.

## CMake and CTest

For an already prepared **native-test** dependency prefix (configure
`src/third_party` with `-DSPDL_BUILD_TESTS=ON -DBUILD_TESTING=OFF` so glog is
shared):

```bash
cmake -S src -B build/native-tests -G Ninja \
  -DCMAKE_PREFIX_PATH=/path/to/dependency/prefix \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DSPDL_BUILD_PYTHON_BINDING=OFF \
  -DSPDL_BUILD_TESTS=ON -DSPDL_USE_FFMPEG_VERSION=8 \
  -DSPDL_TEST_FFMPEG_ROOT="$CONDA_PREFIX"
cmake --build build/native-tests --target libspdl_tests --parallel 2
ctest --test-dir build/native-tests --output-on-failure --no-tests=error
```

Each built FFmpeg variant has its own GoogleTest executable and prefixed CTest
names (for example, `ffmpeg8.SmokeTest.DecodeVideo`). The aggregate
`libspdl_tests` target builds the test executables and media fixtures. The
ordinary GoogleTest filters and CTest `-R`/`-L` filters remain available. CTest
sets a 120-second timeout per test; the build script fails if no tests are found.

## OSS CI scope

`.github/workflows/cpp-tests.yml` runs the CPU/core tests on Linux x86_64 and
macOS arm64 against FFmpeg 8 and 9. It is separate from the Python/wheel matrix
and uploads CTest XML, failure logs and configure diagnostics. Existing
Linux-only procfs and `/dev/full` tests keep their platform-specific skips on
macOS. CUDA tests require a GPU runner and are not enabled by this workflow.
