#!/usr/bin/env bash

env | grep SPDL || true

set -xe

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
