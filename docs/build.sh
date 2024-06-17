#!/usr/bin/env bash

set -eux

cd "$(dirname "$0")"
rm -rf _build source/generated

mkdir -p _build/breathe/doxygen/libspdl/xml/

python3 -m sphinx.cmd.build -v source _build
