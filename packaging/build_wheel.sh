#!/usr/bin/env bash

set -eux

python_version="${1}"

python="/opt/python/${python_version}/bin/python3"

export PATH="/opt/python/${python_version}/bin/:${PATH}"

rm -rf build

"${python}" -m pip install -U cmake ninja
"${python}" -m build --no-isolation --wheel
