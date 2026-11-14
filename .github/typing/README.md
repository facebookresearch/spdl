# Python type checking

The lint workflow runs blocking Pyrefly checks independently of the optional
mypy reference hook. Neither native builds nor CUDA initialization are needed:
Pyrefly reads the checked-in `.pyi` files.

```bash
python -m pip install -r .github/typing/requirements.txt
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pyrefly check --config .github/typing/pyrefly312.toml
pyrefly check --config pyrefly.toml
```

The profiles follow the shared fbcode targets: Python 3.12 for pipeline code
and dataloader/pipeline tests; Python 3.14 for dataloader/source/IO libraries and
IO/CUDA tests. The interpreter used to install dependencies can be Python 3.12;
the configured version controls static analysis. The deprecated autoresearch
application and Meta-only modules are not in this initial shared-code gate.

Both profiles use the Buck checker's compatibility settings: permissive ignore
comments, no checking of wholly unannotated function bodies, return inference
for annotated functions, generated-code handling, ignored deprecation
warnings and informational unused ignores. Explicit import roots prevent
checkout-layout heuristics from changing resolution. Only the three optional
initialization modules omitted from the public distribution may be missing;
other missing imports and type errors fail CI. There is no error baseline,
retry, `continue-on-error`, or `|| true`.

Configuration parity does not mean identical build graphs: the public Pyrefly
release and its bundled typeshed are pinned here, while fbcode uses its own
DotSlash release and dependency manifests. PyTorch's public CPU package also
isn't the internal source build. Keep these pins and the compatibility settings
under review when the internal toolchain changes.
