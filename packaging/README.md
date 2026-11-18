# SPDL packaging

The root `spdl` package is a metapackage. `spdl_core` contains the pipeline and
its Rust buffer backend; `spdl_io` builds the separate media extensions with
CMake. Distribution names, release versions (`VERSION`) and Python runtime
dependencies are unchanged.

## Native pipeline backend

The core build uses setuptools-rust to build `spdl.pipeline.lib._native` from
`src/spdl/pipeline/lib/lib.rs`, through the existing core `src` symlink. The
Rust binding implements buffer exports, allocator ownership and atomic release
tickets, plus the hello example. There is no second implementation or release.

The Python interfaces `spdl.pipeline.lib._buffer` and
`spdl.pipeline.lib._libspdl_pipeline` are importable before Rust is built.
Ordinary pipeline execution remains available without native code; storage
exports require the Rust backend. There is no legacy C++ buffer implementation
or unsafe Python-finalizer substitute. A broken binary keeps its original error.

Default builds require Rust and produce interpreter/platform-specific wheels.
A deliberately Python-only build, without running Rust, is available:

```sh
SPDL_BUILD_PIPELINE_EXTENSION=0 uv build --package spdl_core --wheel
```

That emits a `py3-none-any` wheel with the Python interfaces but no binary.
Installing a prebuilt native wheel never requires Rust. setuptools-rust is
build-only. Use clean build directories when switching native/no-native modes.

## Dependencies and supported builds

The OSS Rust crate uses only `pyo3-ffi=0.30.1` and Rust's standard library.
The locked third-party graph consists of `pyo3-ffi`, `libc`, `pyo3-build-config`
and `target-lexicon`; no high-level PyO3 wrappers or convenience crates are
needed. Their declared licenses are listed on the documentation's installation
page. A neutral build cfg allows the same source to use the existing PyO3 FFI
reexport in fbcode without registering or upgrading a global dependency.

Native Python 3.15/3.15t builds are supported without unsafe version, ABI or GIL
forcing. The wheel jobs set up Rust 1.85.1 and test installed core wheels for
correct tags, lazy loading, real buffer ownership, ordinary pipeline execution
and physical binary absence. SPDL's documented Python minimum remains 3.11;
existing `>=3.10` package metadata is unchanged.

Core sdists include `VERSION`, Cargo metadata/lockfile, all Rust sources, the
Python interfaces and stubs. `uv build --package spdl_core --sdist` produces a
self-contained source archive that rebuilds without repository symlinks. Cargo
fetches locked dependencies normally; the archive does not vendor a toolchain.
Rust sources are omitted from native wheels, and the core wheel contains no IO
or media binding.
