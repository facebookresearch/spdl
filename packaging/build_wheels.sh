#!/usr/bin/env bash

set -eux

cd "$(dirname "$0")"

version="$(cat VERSION)"

uv build --all-packages --wheel

twine check --strict dist/*${version}*.whl

# twine upload dist/spdl-${version}-* --repository spdl
# twine upload dist/spdl_io-${version}-* --repository spdl-io
# twine upload dist/spdl_core-${version}-* --repository spdl-core
