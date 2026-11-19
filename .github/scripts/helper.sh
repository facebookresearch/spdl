#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Source from the repository root; helpers do not change the caller's shell options.
retry_with_backoff() {
    local attempt status delay
    for attempt in 1 2 3 4; do
        if "$@"; then
            return 0
        else
            status=$?
        fi
        if (( attempt == 4 )); then
            echo "Command failed after ${attempt} attempts: $*" >&2
            return "${status}"
        fi
        delay=$((5 << (attempt - 1)))
        echo "Attempt ${attempt} failed; retrying in ${delay}s: $*" >&2
        sleep "${delay}" || return $?
    done
}

# Optional downloader: curl (default) or wget; retain each caller's retry policy.
install_uv() (
    local installer="" status downloader=${1:-curl}
    trap 'status=$?; set +e; rm -f "${installer}"; exit "${status}"' EXIT
    installer=$(mktemp) || return $?
    case "${downloader}" in
        curl) curl -LsSf -o "${installer}" https://astral.sh/uv/install.sh || return $? ;;
        wget) wget -qO "${installer}" https://astral.sh/uv/install.sh || return $? ;;
        *) echo "Unsupported UV downloader: ${downloader}" >&2; return 2 ;;
    esac
    sh "${installer}"
)

# Arguments: Python version, free-threaded marker ("ft" or empty).
setup_build_python() {
    local python_version=$1 free_threaded=${2:-}
    if [[ "${free_threaded}" == ft ]]; then
        python_version="${python_version}t"
    fi
    uv --version || return $?
    uv python pin "${python_version}" || return $?
    uv python list --only-installed || return $?
    uv venv
}

# Arguments: install directory, CUDA version, additional installer options.
configure_cuda_toolkit() {
    local cuda_dir=$1 cuda_version=$2
    shift 2
    .github/scripts/install_cuda_toolkit.py \
        --base-dir "${cuda_dir}" --cuda-version "${cuda_version}" "$@" || return $?
    if [[ ! -d "${cuda_dir}/lib64" ]]; then
        ln -s "${cuda_dir}/lib" "${cuda_dir}/lib64" || return $?
    fi
}

# Arguments: wheel directory, full installer command (including any pip options).
install_wheel_artifacts() (
    local wheel_dir=$1
    shift
    shopt -s nullglob dotglob
    local wheels=("${wheel_dir}"/*.whl)
    if [[ -z "${wheels[0]:-}" ]]; then
        echo "No wheel artifacts found in ${wheel_dir}." >&2
        return 2
    fi
    "$@" "${wheels[@]}"
)

# Arguments: wheel directory, optional Python executable for both check commands.
# Without an interpreter override, preserve the caller's PATH-selected pip/twine.
check_wheel_artifacts() (
    local wheel_dir=$1
    local pip_command=(pip) twine_command=(twine)
    if (( $# > 1 )); then
        pip_command=("$2" -m pip)
        twine_command=("$2" -m twine)
    fi
    shopt -s nullglob dotglob
    local wheels=("${wheel_dir}"/*.whl)
    if [[ -z "${wheels[0]:-}" ]]; then
        echo "No wheel artifacts found in ${wheel_dir}." >&2
        return 2
    fi
    "${pip_command[@]}" install -r ./packaging/requirements.txt || return $?
    "${twine_command[@]}" check --strict "${wheels[@]}"
)

# Arguments: free-threaded marker, explicit Conda install options/environment.
prepare_conda_test_python() {
    local free_threaded=$1
    shift
    local packages=(pip)
    if [[ "${free_threaded}" == ft ]]; then
        packages+=(python-freethreading)
    fi
    conda install "$@" "${packages[@]}"
}

# Arguments: worker concurrency, test suites and any extra pytest options.
# Keep the job-level timeout and platform-specific suites in the workflow.
# Thread-based timeouts work with xdist and on Windows, and dump thread stacks.
run_python_tests() {
    local concurrency=$1
    shift
    pytest -v -n "${concurrency}" \
        --timeout=120 --timeout-method=thread \
        --reruns=2 --reruns-delay=1 --rerun-show-tracebacks -ra "$@"
}

print_python_environment() {
    echo "=== which/where ==="
    command -v python python3 pip || return $?
    python -c "import sys; print('exe=', sys.executable); print('prefix=', sys.prefix); print('path=', sys.path)" || return $?
    python3 -c "import sys; print('exe=', sys.executable); print('prefix=', sys.prefix); print('path=', sys.path)" || return $?
    echo "=== conda env ==="
    echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
    # setup-miniconda can install under HOME when the runner has no bundled Conda.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        ls -ld "${CONDA_PREFIX}" "${CONDA_PREFIX}"/lib/python*/site-packages || return $?
    fi
    echo "=== pip view ==="
    pip -V || return $?
    pip config list -v || return $?
    python -m pip show pygments || true
    python3 -m pip show pygments || true
}
