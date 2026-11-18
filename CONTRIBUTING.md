# Contributing to spdl
We want to make contributing to this project as easy and transparent as
possible.

## Checks

Install Python 3.12 first; pre-commit creates a cached environment but does not
install Python itself. Set up the contributor checks with:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run pyrefly --all-files
pre-commit run mypy --all-files
```

The first checker runs install their pinned CPU typing dependencies automatically;
later runs reuse the caches. No manual Torch installation or native build is
needed. Pyrefly checks its full-project profile, while mypy checks the entire
`src/spdl` directory through the configuration in `pyproject.toml`; tests and
examples are outside mypy's scope. A change under `src/spdl` or to the mypy
configuration triggers this full source check, even in diff-based CI.
`pre-commit run mypy --all-files` also runs the full source check locally.
Both target Python 3.14 and use a Python 3.12 host for the
cached checker environments and dependency wheels. The mypy hook remains
advisory, so inspect its diagnostics rather than treating its `Passed` label as
proof of zero errors. This does not change SPDL's Python 3.11 minimum runtime
support. The pinned Torch wheels support Linux/Windows
CPU installations and Apple Silicon Macs with macOS 14 or newer; there is no
matching Python 3.12 Intel Mac wheel.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to spdl, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
