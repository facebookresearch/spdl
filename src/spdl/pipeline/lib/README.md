# Optional Rust pipeline backend

The `spdl.pipeline.lib` package contains `_native`, the private Rust binary
bundled with `spdl_core`. The importable
Python interfaces `_buffer` and `_libspdl_pipeline` load it only when a native
operation is used. Ordinary pipeline imports and execution do not require a
built binary. Native storage exports fail explicitly when it is absent; there
is no legacy C++ backend or wrapper-finalizer substitute.

The Rust buffer exporter retains the actual `Py_buffer` or allocator owner
until the last native alias releases it. An atomic ticket transfers ownership
from setup to native storage; the destructor releases backing storage before
calling Python lease bookkeeping. Module state and heap types are isolated
per interpreter. The raw CPython bindings use `pyo3-ffi` and the Rust standard
library; dependency and declared-license details are on the installation page.

## Example and builds

After installing a native wheel, run the repository's buffer ownership example:

```bash
python examples/native_buffer.py
```

It retains a slice after releasing the original memoryview, then releases the
storage callback only after the final alias is gone. In fbcode run
`buck2 run fbcode//spdl/examples:native_buffer`.

Standalone source build from the repository root, with Rust installed:

```bash
python -m pip wheel ./packaging/spdl_core --no-deps -w dist
```

A deliberately Python-only build skips Rust:

```bash
SPDL_BUILD_PIPELINE_EXTENSION=0 python -m pip wheel ./packaging/spdl_core --no-deps -w dist
```

Both Python interface modules remain importable in that build, but native
features require the binding. Native builds are enabled by default and do not
silently ignore failures. OSS uses `pyo3-ffi` 0.30.1, including native Python
3.15/3.15t support, without unsafe version or GIL overrides. The same source
uses the build system's existing PyO3 FFI reexport in fbcode. This does not
change the package's Python metadata or repair unrelated compatibility gaps.

The `.pyi` files provide the static contracts without loading the binary.
Cargo builds the standalone distribution; fbcode uses `rust_python_extension`.
