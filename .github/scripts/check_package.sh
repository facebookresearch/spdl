#!/usr/bin/env bash

# Used by Windows CI build jobs.

pip install -r ./packaging/requirements.txt
twine check --strict ./package/*.whl
