#!/usr/bin/env bash

# Used by Windows CI build jobs.
set -e

source .github/scripts/helper.sh
check_wheel_artifacts ./package
