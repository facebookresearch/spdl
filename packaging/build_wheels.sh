#!/usr/bin/env bash

set -eux

cd "$(dirname "$0")"

version="$(cat VERSION)"

uv build --all-packages --wheel

case "$(uname -sr)" in
    Linux*)
        echo "Fixing the wheel."
        for whl in dist/*.whl; do
            ./repair_wheel.sh "${whl}"
        done
    ;;
esac

# twine check --strict dist/*${version}*.whl
# twine upload dist/spdl-${version}-* --repository spdl
# twine upload dist/spdl_io-${version}-* --repository spdl-io
# twine upload dist/spdl_core-${version}-* --repository spdl-core
